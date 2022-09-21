"""Test the performance of models on the Qin data."""
from enum import Enum
from pathlib import Path

from tensorflow.keras.models import Model

from .data.io import QinGraphData, QinDatasets
from .gnn import QinGNN, CoarseGNN

class Experiment:
    """Train a model on the Qin data, then report the results."""
    def __init__(self, model: Model, name: str) -> None:
        self.model = model
        self.name = name
    
    class Paths(Enum):
        """Contains the paths that the model files are stored in."""
        model: Path(self.name)


    def train(self):
        """Train the model, reporting data via tensorboard."""
        ...