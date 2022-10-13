"""Data loading and preprocessing utilities."""
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import KFold, train_test_split
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.transforms import LayerPreprocess

from .featurise.ecfp import ECFPCountFeaturiser, SMILESHashes
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

    def cv_indexes(
        self, num_folds: int = 10, random_seed: int = RANDOM_SEED
    ) -> List[Tuple[List[int], List[int]]]:
        """Get the list of indexes in each fold of a K-fold cross-validation split.

        Args:
            num_folds: The number of folds.
            random_seed: The random state to use when shuffling the data.

        Returns:
            A list of ``(train_indexes, test_indexes)`` for each fold.

        """
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
        return list(kf.split(self.df))

    def train_test_data(
        self, fold: int, num_folds: int = 10, random_seed: int = RANDOM_SEED
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train and test slices for a given fold."""
        train_idxs, test_idxs = self.cv_indexes(
            num_folds=num_folds, random_seed=random_seed
        )[fold]
        return self.df.iloc[train_idxs], self.df.iloc[test_idxs]


class QinDataLoader(ABC):
    """Handle reading Qin datasets from file."""

    def __init__(self, dataset: QinDatasets) -> None:
        """Load data and find train/test indexes.

        Args:
            dataset: Which Qin dataset to load.

        """
        self.df = pd.read_csv(dataset.value, header=0, index_col=0)
        self.df["Molecules"] = [MolFromSmiles(smiles) for smiles in self.df["smiles"]]
        self.test_idxs = np.where(self.df["traintest"] == "test")[0]
        self.train_idxs = np.where(self.df["traintest"] == "train")[0]
        self.optim_idxs, self.val_idxs = train_test_split(self.train_idxs, train_size=0.9, random_state=2022)


class QinECFPData(QinDataLoader):
    """Handle reading Qin datasets from file and featurising with ECFP fingerprints."""

    def __init__(self, dataset: QinDatasets, hash_file: Optional[Path] = None) -> None:
        """Load data and initialise featuriser.

        Args:
            dataset: Which Qin dataset to load.
            hash_file: Where to save/load hash data to.

        """
        super().__init__(dataset)

        smiles_hashes = None
        save_hashes = False

        if hash_file is not None:
            if hash_file.exists():
                smiles_hashes = SMILESHashes.load(hash_file)
            else:
                save_hashes = True

        if smiles_hashes is None:
            smiles_hashes = SMILESHashes()
        
        self.smiles_hashes: SMILESHashes = smiles_hashes

        self.featuriser = ECFPCountFeaturiser(self.smiles_hashes)

        self.fingerprints = self.featuriser.featurise_molecules(
            list(self.df["Molecules"]), 2
        )
        if save_hashes:
            self.featuriser.smiles_hashes.save(hash_file)

    def get_at_idxs(self, indexes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the fingerprints and target values for the given indexes."""
        return self.fingerprints[indexes, :], self.df.exp.loc[indexes]

    @property
    def train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get numpy arrays of training data fingerprints and targets."""
        return self.get_at_idxs(self.train_idxs)

    @property
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get numpy arrays of test data fingerprints and targets."""
        return self.get_at_idxs(self.test_idxs)

    @property
    def all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get numpy arrays of all data fingerprints and targets."""
        return self.fingerprints, self.df.exp.to_numpy()


class QinGraphData(QinDataLoader):
    """Handle reading Qin datasets from file and splitting into train and test subsets."""

    def __init__(
        self,
        dataset: QinDatasets,
        mol_featuriser: MolNodeFeaturizer = MolNodeFeaturizer(),
        preprocess: Optional[LayerPreprocess] = None,
    ) -> None:
        """Load data and initialise featuriser.

        Args:
            dataset: Which dataset to load.
            mol_featuriser: The molecular featuriser to use. This is important for consistency with featurising, e.g. one hot encoding.

        """
        super().__init__(dataset)

        graphs = mols_to_graph(
            list(self.df["Molecules"]), mol_featuriser, list(self.df["exp"])
        )
        self.graphs = (
            list(map(preprocess, graphs)) if preprocess is not None else graphs
        )

    class Subset(Dataset):
        """Handle graph data subsets."""

        def __init__(self, graphs: List[Graph]):
            """Store graphs."""
            self.graphs = graphs
            super().__init__()

        def read(self) -> List[Graph]:
            """Return the graphs for this subset."""
            return self.graphs

    @property
    def train_dataset(self):
        """Get the training dataset."""
        train_graphs = [self.graphs[i] for i in self.train_idxs]
        return self.Subset(train_graphs)
    
    @property
    def optim_dataset(self):
        """Get the optimisation dataset."""
        optim_graphs = [self.graphs[i] for i in self.optim_idxs]
        return self.Subset(optim_graphs)

    @property
    def val_dataset(self):
        """Get the validation dataset."""
        val_graphs = [self.graphs[i] for i in self.val_idxs]
        return self.Subset(val_graphs)

    @property
    def test_dataset(self):
        """Get the test dataset."""
        test_graphs = [self.graphs[i] for i in self.test_idxs]
        return self.Subset(test_graphs)

    @property
    def all_dataset(self):
        """Get the full dataset."""
        return self.Subset(self.graphs)

    @property
    def train_loader(self):
        """Get the training data loader."""
        return DisjointLoader(self.train_dataset)

    @property
    def optim_loader(self):
        """Get the optimisation data loader."""
        return DisjointLoader(self.optim_dataset)

    @property
    def val_loader(self):
        """Get the validation data loader."""
        return DisjointLoader(self.val_dataset)

    @property
    def test_loader(self):
        """Get the test data loader."""
        return DisjointLoader(self.test_dataset)

    @property
    def all_loader(self):
        """Get the full data loader."""
        return DisjointLoader(self.all_dataset, shuffle=False)
