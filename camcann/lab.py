"""Test the performance of models on the Qin data."""
from argparse import ArgumentParser
from ast import parse
from datetime import datetime
from pathlib import Path
from typing import Type

import pandas as pd
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.models import Model

from .data.io import QinDatasets, QinECFPData, QinGraphData
from .gnn import CoarseGNN, QinGNN
from .linear import LinearECFPModel, RidgeResults


class GraphExperiment:
    """Train a model on the Qin data, then report the results."""

    def __init__(
        self, model: Type[Model], dataset: QinDatasets, results_path: Path
    ) -> None:
        """Initialize the model and the datasets."""
        self.model: Model = model()
        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=[RootMeanSquaredError(), MeanAbsoluteError()],
        )
        if isinstance(self.model, QinGNN):
            preprocess = LayerPreprocess(GCNConv)
        else:
            preprocess = None
        self.graph_data = QinGraphData(dataset, preprocess=preprocess)

        print("First 10 graphs:")
        print(self.graph_data.graphs[:10])
        first_graph = self.graph_data.graphs[0]
        print("First graph's data:")
        print(f"{first_graph.x=}")
        print(f"{first_graph.a=}")

        self.results_path = results_path
        self.model_path = results_path / "model"
        self.predict_path = results_path / "predictions.csv"
        self.tb_dir = results_path / "logs"
        self.metrics_path = results_path / "metrics.csv"
        for path in [self.results_path, self.tb_dir]:
            if not path.exists():
                path.mkdir()

    @property
    def tb_run_dir(self) -> Path:
        """Get a new tensorboard log directory for a current run."""
        return self.tb_dir / str(datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))

    def train(self, epochs: int):
        """Train the model, reporting data via tensorboard."""
        loader = self.graph_data.train_loader

        callbacks = []
        # if self.patience:
        #     es_callback = EarlyStopping(
        #         min_delta=self.min_delta,
        #         patience=self.patience,
        #         restore_best_weights=True,
        #     )
        #     callbacks.append(es_callback)
        callbacks.append(TensorBoard(log_dir=self.tb_run_dir))

        self.model.fit(
            loader.load(),
            steps_per_epoch=loader.steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
        )
        self.model.save(self.model_path)

    def _make_pred_df(self, predictions):
        """Make a DataFrame of predicted CMCs."""
        return pd.DataFrame(
            {
                "smiles": self.graph_data.df.smiles,
                "exp": self.graph_data.df.exp,
                "qin": self.graph_data.df.pred,
                "pred": predictions,
                "traintest": self.graph_data.df.traintest,
            }
        )

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
        )
        self._make_pred_df(predictions).to_csv(self.predict_path)


class ECFPExperiment:
    """Train and evaluate a simple, linear ECFP model."""

    def __init__(self, dataset: QinDatasets, results_path: Path) -> None:
        """Load dataset and initialise featuriser."""
        self.results_path = results_path
        self.hash_path = results_path / "hashes.csv"
        self.predict_path = results_path / "predictions.csv"
        self.metrics_path = results_path / "metrics.csv"
        if not self.results_path.exists():
            self.results_path.mkdir()

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
        self, num_low_freq: int, num_low_import: int, ridge_results: RidgeResults
    ) -> pd.Series:
        """Get metrics as a pandas series for writing to disk."""
        return pd.Series(
            {
                "num_low_freq": num_low_freq,
                "num_low_import": num_low_import,
                "best_train_rmse": ridge_results.best_rmse,
                "best_alpha": ridge_results.alpha,
                "test_rmse": ridge_results.test_rmse,
            }
        )

    def train_test(self):
        """Run training and testing routine."""
        print("Removing low frequency subgraphs...")
        num_low_freq = self.model.remove_low_freq_subgraphs()
        print(f"{num_low_freq} subgraphs removed.")
        print("Doing elastic net feature selection...")
        num_low_import = self.model.elastic_feature_select()
        print(f"{num_low_import} subgraphs removed.")
        print("Fitting ridge models...")
        ridge_results = self.model.ridge_model_train_test()
        print(ridge_results)

        # Write results
        self.model.smiles_hashes.save(self.hash_path)
        predictions = self.model.predict(self.featuriser.all_data[0])
        results_df = self._make_pred_df(predictions)
        results_df.to_csv(self.predict_path)
        self._metrics_series(num_low_freq, num_low_import, ridge_results).to_csv(
            self.metrics_path
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    dataset_map = {
        "Nonionics": QinDatasets.QIN_NONIONICS_RESULTS,
        "All": QinDatasets.QIN_ALL_RESULTS,
    }
    model_map = {
        "QinModel": QinGNN,
        "CoarseModel": CoarseGNN,
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
    parser.add_argument("epochs", type=int, help="The number of epochs to train.")
    args = parser.parse_args()

    dataset = dataset_map[args.dataset]
    model = model_map[args.model]

    results_path = Path(".") / args.name

    if model is LinearECFPModel:
        exp = ECFPExperiment(dataset, results_path=results_path)
        exp.train_test()
    else:
        exp = GraphExperiment(model, dataset, results_path=results_path)
        exp.train(args.epochs)
        exp.test()
