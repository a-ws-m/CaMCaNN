"""Test the performance of models on the Qin data."""
from argparse import ArgumentParser
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, Tuple

import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import Model, load_model

from .data.io import (
    Datasets,
    RANDOM_SEED,
    QinDatasets,
    ECFPData,
    QinGraphData,
    get_nist_data,
    get_nist_and_qin,
)
from .gnn import build_gnn
from .linear import LinearECFPModel, LinearResults
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
        self,
        model,
        dataset: Union[Datasets, QinDatasets],
        results_path: Path,
        debug: bool = False,
    ) -> None:
        """Initialise paths."""
        self.results_path = results_path
        self.model_path = results_path / "model"

        self.predict_path = results_path / "predictions.csv"
        self.uq_predict_path = results_path / "uq_predictions.csv"

        self.nist_pred_path = results_path / "nist_predictions.csv"
        self.nist_res_path = results_path / "nist_results.csv"

        self.metrics_path = results_path / "metrics.csv"
        self.uq_train_metrics_path = results_path / "uq_train_metrics.csv"
        self.uq_val_metrics_path = results_path / "uq_val_metrics.csv"
        self.uq_test_metrics_path = results_path / "uq_test_metrics.csv"

        self.uq_nist_metrics_path = results_path / "uq_nist_metrics.csv"
        self.uq_nist_pred_path = results_path / "uq_nist_pred.csv"

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

        self.kpca_file = self.results_path / "kernel_components.csv"

        self.kernel_file = self.results_path / "full_kernel.csv"

        self.gp_param_file = self.model_path / "gp_params.json"

        self.hypermodel = hypermodel
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
            with self.best_hp_file.open("r") as f:
                self.best_hps_dict = json.load(f)

            hps = keras_tuner.HyperParameters()
            self.hypermodel(hps)
            hps.values = self.best_hps_dict
            self.model = self.tuner.hypermodel.build(hps)

            loaded_model = load_model(self.model_path)

            train_data = self.graph_data.train_loader_no_shuffle
            self.model.predict(train_data.load(), steps=train_data.steps_per_epoch)
            loaded_model.predict(train_data.load(), steps=train_data.steps_per_epoch)

            for latent_layer, buffer in zip(self.model.layers, loaded_model.layers):
                latent_layer.set_weights(buffer.get_weights())

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

        self.best_hps_dict = best_hp.values
        print("Best hyperparameters:")
        print(self.best_hps_dict)

        with self.best_hp_file.open("w") as f:
            json.dump(self.best_hps_dict, f)

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
        nist_data, nist_df = get_nist_data(
            self.graph_data.mol_featuriser, preprocess=LayerPreprocess(GCNConv)
        )

        nist_metrics = loaded_model.evaluate(
            nist_data.load(), steps=nist_data.steps_per_epoch, return_dict=True
        )
        nist_predictions = loaded_model.predict(
            nist_data.load(), steps=nist_data.steps_per_epoch
        )

        with (self.nist_res_path).open("w") as f:
            json.dump(nist_metrics, f)

        # nist_df["pred"] = None
        # nist_df["pred"][nist_df["Convertable"]] = nist_predictions.flatten()
        nist_df["pred"] = nist_predictions.flatten()
        nist_df.to_csv(self.nist_pred_path, columns=["SMILES", "log CMC", "pred"])

        return nist_metrics

    def train_uq(
        self,
        with_scaler: bool = True,
        linear_mean_fn: bool = False,
        retrain: bool = False,
    ):
        """Train and test the uncertainty quantified model."""
        hps = keras_tuner.HyperParameters()
        self.hypermodel(hps)
        hps.values = self.best_hps_dict
        latent_model = self.hypermodel(hps, latent_model=True)

        loaded_model = load_model(self.model_path)

        load_gp_params = self.gp_param_file.exists() and not retrain
        param_path = self.gp_param_file if load_gp_params else None

        self.uq_model = GraphGPProcess(
            latent_model,
            self.graph_data,
            loaded_model,
            with_scaler,
            linear_mean_fn,
            param_path,
        )
        self.uq_model.save_model(self.gp_param_file)

        train_metrics = self.uq_model.evaluate(self.graph_data.optim_loader_no_shuffle)
        val_metrics = self.uq_model.evaluate(self.graph_data.val_loader)
        test_metrics = self.uq_model.evaluate(self.graph_data.test_loader)

        pd.Series(train_metrics).to_csv(self.uq_train_metrics_path)
        pd.Series(val_metrics).to_csv(self.uq_val_metrics_path)
        pd.Series(test_metrics).to_csv(self.uq_test_metrics_path)

        means, stddevs = self.uq_model.predict(self.graph_data.all_loader)
        self._make_pred_df(means, stddevs).to_csv(self.uq_predict_path)

        nist_data, nist_df = get_nist_data(
            self.graph_data.mol_featuriser, preprocess=LayerPreprocess(GCNConv)
        )
        nist_means, nist_stddevs = self.uq_model.predict(nist_data)
        nist_df["pred"] = nist_means.flatten()
        nist_df["stddev"] = nist_stddevs.flatten()
        nist_df.to_csv(self.uq_nist_pred_path)

        nist_data, nist_df = get_nist_data(
            self.graph_data.mol_featuriser, preprocess=LayerPreprocess(GCNConv)
        )
        nist_metrics = self.uq_model.evaluate(nist_data)
        pd.Series(nist_metrics).to_csv(self.uq_nist_metrics_path)

    def kpca(self, ndim: int = 2):
        """Perform KPCA on NIST and Qin data."""
        kernel_matrix, combined_data = self.pairwise()

        components = KernelPCA(ndim, kernel="precomputed").fit_transform(kernel_matrix)

        for dim in range(ndim):
            combined_data[f"Component {dim+1}"] = components[:, dim]

        combined_data.to_csv(self.kpca_file)
    
    def pairwise(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute the kernel matrix on the NIST and Qin data."""
        combined_loader, combined_data = get_nist_and_qin(
            self.graph_data.mol_featuriser, preprocess=LayerPreprocess(GCNConv)
        )
        kernel_matrix = self.uq_model.pairwise_matrix(combined_loader)

        for j in range(kernel_matrix.shape[1]):
            combined_data[f"K{j}"] = kernel_matrix[:, j]
        
        combined_data.to_csv(self.kernel_file)
        return kernel_matrix, combined_data

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

    def __init__(
        self, dataset: Union[Datasets, QinDatasets], results_path: Path
    ) -> None:
        """Load dataset and initialise featuriser."""
        super().__init__(model, dataset, results_path)

        self.hash_path = results_path / "hashes.csv"

        self.featuriser = ECFPData(dataset, self.hash_path)

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
        self, num_low_freq: int, elastic_results: LinearResults
    ) -> pd.Series:
        """Get metrics as a pandas series for writing to disk."""
        metric_series = asdict(elastic_results)
        metric_series.pop("coefs")
        metric_series["num_low_freq"] = num_low_freq
        return pd.Series(metric_series)

    def train_test(self):
        """Run training and testing routine."""
        print("Removing low frequency subgraphs...")
        num_low_freq = self.model.remove_low_freq_subgraphs()
        print(f"{num_low_freq} subgraphs removed.")
        print("Doing elastic net feature selection...")
        elastic_results = self.model.elastic_feature_select()

        # Write results
        self.model.smiles_hashes.save(self.hash_path)
        predictions = self.model.predict(self.featuriser.all_data[0])
        results_df = self._make_pred_df(predictions)
        results_df.to_csv(self.predict_path)

        metrics = self._metrics_series(num_low_freq, elastic_results)
        print(metrics)
        metrics.to_csv(self.metrics_path)

    def test_nist(self):
        """Test against the NIST data."""
        nist_data = ECFPData(Datasets.NIST_NEW, self.hash_path)

        fps, targets = nist_data.all_data
        nist_predictions = self.model.predict(fps)

        nist_df_copy = nist_data.df.copy(deep=True)
        nist_df_copy["pred"] = nist_predictions

        nist_df_copy.to_csv(self.nist_pred_path)

        rmse = mean_squared_error(targets, nist_predictions, squared=False)
        mse = rmse**2
        mae = mean_absolute_error(targets, nist_predictions)
        results = {"mse": mse, "rmse": rmse, "mae": mae}

        with self.nist_res_path.open("w") as f:
            json.dump(results, f)


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
    parser.add_argument(
        "--no-gp-scaler",
        action="store_true",
        help="Don't use a scaler on the latent points for the Gaussian process.",
    )
    parser.add_argument(
        "--lin-mean-fn",
        action="store_true",
        help="Use a linear function for the mean of the Gaussian process. If not set, will use the trained MLP from the NN as the mean function.",
    )
    parser.add_argument(
        "--only-best",
        action="store_true",
        help="For GNN -- don't perform search, only train the best model.",
    )
    parser.add_argument(
        "--test-nist",
        action="store_true",
        help="Test saved model on NIST anionics data.",
    )
    parser.add_argument(
        "--kpca",
        type=int,
        help="Compute N kernel principal components on Qin and NIST data after training UQ.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Compute just the learned pariwise kernel on all of the data."
    )

    args = parser.parse_args()

    if args.and_uq and args.just_uq:
        raise ValueError("Cannot set both `--and-uq` and `--just-uq` flags.")
    if (args.kpca or args.pairwise) and not (args.and_uq or args.just_uq):
        raise ValueError(
            "Must train UQ using `--and-uq` or `--just-uq` to compute kernel matrix."
        )
    if args.kpca is not None and args.kpca <= 0:
        raise ValueError("KPCA components must be positive.")

    dataset = dataset_map[args.dataset]
    model = model_map[args.model]

    do_uq = args.and_uq or args.just_uq

    results_path = Path(".") / args.name

    if model is LinearECFPModel:
        if do_uq:
            raise NotImplementedError("Cannot use UQ with linear ECFP model.")

        ecfp_exp = ECFPExperiment(dataset, results_path=results_path)
        ecfp_exp.train_test()
        if args.test_nist:
            ecfp_exp.test_nist()
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
                exp.train_uq(
                    with_scaler=not args.no_gp_scaler, linear_mean_fn=args.lin_mean_fn
                )
                if args.kpca:
                    exp.kpca(args.kpca)
                elif args.pairwise:
                    exp.pairwise()
