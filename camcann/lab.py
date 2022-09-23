"""Test the performance of models on the Qin data."""
from argparse import ArgumentParser
from ast import parse
from datetime import datetime
from pathlib import Path
from typing import Type

import pandas as pd
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import (
    MeanAbsoluteError,
    RootMeanSquaredError,
)
from tensorflow.keras.models import Model

from .data.io import QinDatasets, QinECFPData, QinGraphData
from .gnn import CoarseGNN, QinGNN

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
        self.graph_data = QinGraphData(dataset)
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
        self.model_path = results_path / "model"
        self.predict_path = results_path / "predictions.csv"
        self.metrics_path = results_path / "metrics.csv"
        for path in [self.results_path, self.tb_dir]:
            if not path.exists():
                path.mkdir()

        self.featuriser = QinECFPData(dataset)


if __name__ == "__main__":
    parser = ArgumentParser()

    dataset_map = {"Nonionics": QinDatasets.QIN_NONIONICS_RESULTS, "All": QinDatasets.QIN_ALL_RESULTS}
    model_map = {"QinModel": QinGNN, "CoarseModel": CoarseGNN, "ECFPLinear": None}

    parser.add_argument("--model", dest="model", choices=list(model_map.keys()), help="The type of model to create.")
    parser.add_argument("--dataset", dest="dataset", choices=list(dataset_map.keys()), help="The dataset to use.")
    parser.add_argument("--name", dest="name", type=str, help="The name of the model.")
    parser.add_argument("epochs", dest="epochs", type=int, help="The number of epochs to train.")
    args = parser.parse_args()

    dataset = dataset_map[args.dataset]
    model = model_map[args.model]

    exp = GraphExperiment(
        model, dataset, results_path=Path(".") / "test_model"
    )
    exp.train(args.epochs)
    exp.test()
