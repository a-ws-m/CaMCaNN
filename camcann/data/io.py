"""Data loading and preprocessing utilities."""
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import KFold
from spektral.data import Dataset, Graph
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess

from .featurise.graph import MolNodeFeaturizer, mols_to_graph

DATASET_FOLDER = Path(__file__).parent / "datasets"
RANDOM_SEED = 2021

class Datasets(Enum):
    """Available datasets."""

    QIN = DATASET_FOLDER / "qin-data.csv"
    NIST_ANIONICS = DATASET_FOLDER / "nist-anionics.csv"
    QIN_AND_NIST_ANIONICS = DATASET_FOLDER / "merged-data.csv"

class QinDatasets(Enum):
    """Qin datasets split by test and train subsets."""

    QIN_NONIONICS_RESULTS = DATASET_FOLDER / "qin_nonionic_results.csv"
    QIN_ALL_RESULTS = DATASET_FOLDER / "qin_all_results.csv"


class DataReader:
    """Handle reading datasets from disk and preprocessing, plus cross-validation splitting."""

    def __init__(self, dataset: Datasets) -> None:
        """Read data from disk."""
        self.df = pd.read_csv(dataset.value, header=0)
    
    @property
    def cv_indexes(self, num_folds: int = 10, random_seed: int = RANDOM_SEED) -> List[Tuple[List[int], List[int]]]:
        """Get the list of indexes in each fold of a K-fold cross-validation split.
        
        Args:
            num_folds: The number of folds.
            random_seed: The random state to use when shuffling the data.
        
        Returns:
            A list of ``(train_indexes, test_indexes)`` for each fold.
        
        """
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
        return list(kf.split(self.df))
    
    @property
    def train_test_data(self, fold: int, num_folds: int = 10, random_seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train and test slices for a given fold."""
        train_idxs, test_idxs = self.cv_indexes(num_folds=num_folds, random_seed=random_seed)[fold]
        return self.df.iloc[train_idxs], self.df.iloc[test_idxs]

class QinGraphData(Dataset):
    """Handle reading Qin datasets from file and splitting into train and test subsets."""

    def __init__(self, dataset: QinDatasets, read_train: bool=True, mol_featuriser: MolNodeFeaturizer = MolNodeFeaturizer()) -> None:
        """Load data.
        
        Args:
            dataset: Which dataset to load.
            read_train: Whether to read train data. :meth:`read` will load test data if this is ``False``.
            mol_featuriser: The molecular featuriser to use. This is important for consistency with featurising, e.g. one hot encoding.
        
        """
        df = pd.read_csv(dataset.value, header=0, index_col=0)
        df.Molecules = [MolFromSmiles(smiles) for smiles in df.smiles]
        df.Graphs = mols_to_graph(df.Molecules, mol_featuriser, df.exp)
        self.graphs = list(df.Graphs)
        self.apply(LayerPreprocess(GCNConv))

        self.test_idxs, _ = np.where(df.traintest=="test")
        self.train_idxs, _ = np.where(df.traintest=="train")

        self.read_train: bool = read_train

    def read(self) -> List[Graph]:
        """Return the graphs for the selected dataset."""
        return [self.graphs[i] for i in (self.train_idxs if self.read_train else self.test_idxs)]
