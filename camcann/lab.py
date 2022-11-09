"""Test the performance of models on the Qin data."""
from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import Model, load_model

from .data.io import RANDOM_SEED, QinDatasets, QinECFPData, QinGraphData, get_nist_data
from .gnn import build_gnn
from .linear import LinearECFPModel, ElasticResults
from .uq import GraphGPProcess

RANDOM_SEED = 2022


class BaseExperiment:
    """Train a model on the Qin data, potentially with separate UQ, and report results.

    Args:
        model: The type of model to use.
        dataset: Which Qin dataset to use.
        results_path: The directory in which to save results.
        debug: Enable debugging print messages.

    """

    def __init__(
        self, model, dataset: QinDatasets, results_path: Path, debug: bool = False
    ) -> None:
        """Initialise paths."""
        self.results_path = results_path
        self.model_path = results_path / "model"

        self.predict_path = results_path / "predictions.csv"
        self.uq_predict_path = results_path / "uq_predictions.csv"

        self.metrics_path = results_path / "metrics.csv"
        self.uq_train_metrics_path = results_path / "uq_train_metrics.csv"
        self.uq_test_metrics_path = results_path / "uq_test_metrics.csv"

        self.results_path.mkdir(exist_ok=True)

        self.model_type = model
        self.dataset = dataset
        self.debug = debug


class GraphExperiment(BaseExperiment):
    """Train a model on the Qin data, as well as a model with UQ, then report their results.

    Args:
        model: The type of model to use.
        dataset: Which Qin dataset to use.
        results_path: Where to save the model, its predictions and its metrics.
        pretrained: If training the UQ model, this specifies whether there is a pre-existing model in the :attr:`results_path`.

    """

    def __init__(
        self,
        hypermodel: Callable[[Any], Model],
        dataset: QinDatasets,
        results_path: Path,
        debug: bool = False,
        pretrained: bool = False,
    ) -> None:
        """Initialize the model and the datasets."""
        super().__init__(hypermodel, dataset, results_path, debug)

        self.tb_dir = results_path / "logs"
        self.tb_dir.mkdir(exist_ok=True)

        self.tb_search_dir = self.tb_dir / "search-logs"
        self.tb_train_dir = self.tb_dir / "train-best"

        self.best_hp_file = self.results_path / "best_hps.json"

        self.tuner = keras_tuner.Hyperband(
            hypermodel=hypermodel,
            objective="val_loss",
            max_epochs=550,
            seed=2022,
            hyperband_iterations=1,
            directory=str(results_path.absolute()),
            project_name="gnn_search",
        )

        self.graph_data = QinGraphData(dataset, preprocess=LayerPreprocess(GCNConv))

        if pretrained:
            loaded_model = self.tuner.get_best_models()[0]

            train_data = self.graph_data.train_dataset
            # TODO: Fix this using build
            # self.model.predict(train_data.load(), steps=train_data.steps_per_epoch)
            # loaded_model.predict(train_data.load(), steps=train_data.steps_per_epoch)

            # for latent_layer, buffer in zip(self.model.layers, loaded_model.layers):
            #     latent_layer.set_weights(buffer.get_weights())

        if self.debug:
            print("First 10 graphs:")
            print(self.graph_data.graphs[:10])
            first_graph = self.graph_data.graphs[0]
            print("First graph's data:")
            print(f"{first_graph.x=}")
            print(f"{first_graph.a=}")

    def search(self):
        """Search the hyperparameter space, reporting data via tensorboard."""
        loader = self.graph_data.optim_loader

        callbacks = []
        es_callback = EarlyStopping(
            min_delta=0,
            patience=100,
        )
        callbacks.append(es_callback)
        callbacks.append(TensorBoard(log_dir=self.tb_search_dir))

        self.tuner.search(
            loader.load(),
            steps_per_epoch=loader.steps_per_epoch,
            validation_data=self.graph_data.val_loader.load(),
            validation_steps=self.graph_data.val_loader.steps_per_epoch,
            callbacks=callbacks,
        )

    def train_best(self, epochs: int):
        """Train the best hyperparameters on all the data."""
        best_hp = self.tuner.get_best_hyperparameters()[0]

        best_hps_dict = best_hp.get_config()["values"]
        print("Best hyperparameters:")
        print(best_hps_dict)

        with self.best_hp_file.open("w") as f:
            json.dump(best_hps_dict, f)

        self.model = self.tuner.hypermodel.build(best_hp)

        es_callback = EarlyStopping(
            min_delta=0,
            patience=150,
            restore_best_weights=True,
        )
        callbacks = [TensorBoard(log_dir=self.tb_train_dir / "with-val"), es_callback]

        print("Fitting best model with validation...")
        self.model.fit(
            self.graph_data.optim_loader.load(),
            steps_per_epoch=self.graph_data.optim_loader.steps_per_epoch,
            validation_data=self.graph_data.val_loader.load(),
            validation_steps=self.graph_data.val_loader.steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
        )
        self.model.save(self.model_path)

        print("Fine tuning best model on all data...")
        callbacks = [TensorBoard(log_dir=self.tb_train_dir / "fine-tune")]
        self.model.fit(
            self.graph_data.train_loader.load(),
            steps_per_epoch=self.graph_data.train_loader.steps_per_epoch,
            epochs=np.floor_divide(epochs, 10),
            callbacks=callbacks,
        )
        self.model.save(self.model_path)
        print("Done!")

    def test_nist(self) -> Dict[str, float]:
        """Test against the NIST data."""
        loaded_model = load_model(self.model_path)
        nist_data, nist_df = get_nist_data(self.graph_data.mol_featuriser, preprocess=LayerPreprocess(GCNConv))

        nist_metrics = loaded_model.evaluate(nist_data.load(), steps=nist_data.steps_per_epoch, return_dict=True)
        nist_predictions = loaded_model.predict(nist_data.load(), steps=nist_data.steps_per_epoch)

        with (self.results_path / "nist-results.json").open("w") as f:
            json.dump(nist_metrics, f)

        # nist_df["pred"] = None
        # nist_df["pred"][nist_df["Convertable"]] = nist_predictions.flatten()
        nist_df["pred"] = nist_predictions.flatten()
        nist_df.to_csv(self.results_path / "nist-predictions.csv", columns=["SMILES", "log CMC", "pred"])

        return nist_metrics

    def train_uq(self):
        """Train and test the uncertainty quantified model."""
        loaded_model = load_model(self.model_path)
        self.uq_model = GraphGPProcess(
            self.model, self.graph_data.train_loader, loaded_model
        )

        train_metrics = self.uq_model.evaluate(self.graph_data.train_loader)
        test_metrics = self.uq_model.evaluate(self.graph_data.test_loader)

        pd.Series(train_metrics).to_csv(self.uq_train_metrics_path)
        pd.Series(test_metrics).to_csv(self.uq_test_metrics_path)

        means, stddevs = self.uq_model.predict(self.graph_data.all_loader)
        self._make_pred_df(means, stddevs).to_csv(self.uq_predict_path)

    def _make_pred_df(self, predictions, stddevs: Optional[np.ndarray] = None):
        """Make a DataFrame of predicted CMCs."""
        data = {
            "smiles": self.graph_data.df.smiles,
            "exp": self.graph_data.df.exp,
            "qin": self.graph_data.df.pred,
            "pred": predictions,
            "traintest": self.graph_data.df.traintest,
        }

        if stddevs is not None:
            data["stddev"] = stddevs

        return pd.DataFrame(data)

    def test(self):
        """Get test metrics and report predictions for all data."""
        loader = self.graph_data.test_loader

        metrics = self.model.evaluate(
            loader.load(), steps=loader.steps_per_epoch, return_dict=True
        )
        pd.Series(metrics).to_csv(self.metrics_path)

        all_loader = self.graph_data.all_loader
        predictions = self.model.predict(
            all_loader.load(), steps=all_loader.steps_per_epoch
        ).flatten()
        self._make_pred_df(predictions).to_csv(self.predict_path)


class ECFPExperiment(BaseExperiment):
    """Train and evaluate a simple, linear ECFP model."""

    def __init__(self, dataset: QinDatasets, results_path: Path) -> None:
        """Load dataset and initialise featuriser."""
        super().__init__(model, dataset, results_path)

        self.hash_path = results_path / "hashes.csv"

        self.featuriser = QinECFPData(dataset)

        train_fps, train_targets = self.featuriser.train_data
        test_fps, test_targets = self.featuriser.test_data

        self.model = LinearECFPModel(
            self.featuriser.smiles_hashes,
            train_fps,
            train_targets,
            test_fps,
            test_targets,
        )

    def _make_pred_df(self, predictions):
        """Make a DataFrame of predicted CMCs."""
        df = self.featuriser.df
        return pd.DataFrame(
            {
                "smiles": df.smiles,
                "exp": df.exp,
                "qin": df.pred,
                "pred": predictions,
                "traintest": df.traintest,
            }
        )

    def _metrics_series(
        self, num_low_freq: int, elastic_results: ElasticResults
    ) -> pd.Series:
        """Get metrics as a pandas series for writing to disk."""
        return pd.Series(
            {
                "num_low_freq": num_low_freq,
                "num_non_negligible": elastic_results.num_non_negligible,
                "train_rmse": elastic_results.train_rmse,
                "best_alpha": elastic_results.alpha,
                "test_rmse": elastic_results.test_rmse,
            }
        )

    def train_test(self):
        """Run training and testing routine."""
        print("Removing low frequency subgraphs...")
        num_low_freq = self.model.remove_low_freq_subgraphs()
        print(f"{num_low_freq} subgraphs removed.")
        print("Doing elastic net feature selection...")
        elastic_results = self.model.elastic_feature_select()
        print(f"{num_low_import} subgraphs removed.")

        # Write results
        self.model.smiles_hashes.save(self.hash_path)
        predictions = self.model.predict(self.featuriser.all_data[0])
        results_df = self._make_pred_df(predictions)
        results_df.to_csv(self.predict_path)
        self._metrics_series(num_low_freq, num_low_import, elastic_results).to_csv(
            self.metrics_path
        )


if __name__ == "__main__":

    # Set random seed
    tf.random.set_seed(RANDOM_SEED)

    parser = ArgumentParser()

    dataset_map = {
        "Nonionics": QinDatasets.QIN_NONIONICS_RESULTS,
        "All": QinDatasets.QIN_ALL_RESULTS,
    }
    model_map = {
        "GNNModel": build_gnn,
        "ECFPLinear": LinearECFPModel,
    }

    parser.add_argument(
        "model",
        choices=list(model_map.keys()),
        help="The type of model to create.",
    )
    parser.add_argument(
        "dataset",
        choices=list(dataset_map.keys()),
        help="The dataset to use.",
    )
    parser.add_argument("name", type=str, help="The name of the model.")
    parser.add_argument(
        "-e", "--epochs", type=int, help="The number of epochs to train."
    )
    parser.add_argument(
        "--and-uq",
        action="store_true",
        help="Re-train the GNN and then the uncertainty quantification model.",
    )
    parser.add_argument(
        "--just-uq", action="store_true", help="Just train the uncertainty quantifier."
    )
    parser.add_argument("--only-best", action="store_true", help="For GNN -- don't perform search, only train the best model.")
    parser.add_argument("--test-nist", action="store_true", help="Test saved model on NIST anionics data.")

    args = parser.parse_args()

    if args.and_uq and args.just_uq:
        raise ValueError("Cannot set both `--and-uq` and `--just-uq` flags.")

    dataset = dataset_map[args.dataset]
    model = model_map[args.model]

    do_uq = args.and_uq or args.just_uq

    results_path = Path(".") / args.name

    if model is LinearECFPModel:
        if do_uq:
            raise NotImplementedError("Cannot use UQ with linear ECFP model.")

        exp: BaseExperiment = ECFPExperiment(dataset, results_path=results_path)
        exp.train_test()
    else:
        pretrained = args.just_uq
        exp = GraphExperiment(
            build_gnn, dataset, results_path=results_path, pretrained=pretrained
        )
        if args.test_nist:
            print(exp.test_nist())
        else:
            if not pretrained:
                if not args.only_best:
                    exp.search()
                exp.train_best(args.epochs)
                exp.test()
            if do_uq:
                exp.train_uq()
