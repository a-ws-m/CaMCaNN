"""Test the performance of models on the Qin data."""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Type

import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
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
        self.model = model()
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
            loader.load(), steps_per_epoch=loader.steps_per_epoch, return_dict=True
        )
        pd.DataFrame(metrics).to_csv(self.metrics_path)

        all_loader = self.graph_data.all_loader
        predictions = self.model.predict(
            all_loader.load(), steps_per_epoch=all_loader.steps_per_epoch
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
    exp = GraphExperiment(
        QinGNN, QinDatasets.QIN_NONIONICS_RESULTS, results_path=Path(".") / "test_model"
    )
    exp.train(500)
    exp.test()
