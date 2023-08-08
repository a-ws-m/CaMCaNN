"""Data loading and preprocessing utilities."""
from abc import ABC
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union
import itertools

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import KFold, train_test_split, RepeatedStratifiedKFold
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.transforms import LayerPreprocess

from .featurise.ecfp import ECFPCountFeaturiser, SMILESHashes
from .featurise.graph import MolNodeFeaturizer, mols_to_graph

DATASET_FOLDER = Path(__file__).parent / "datasets"
RANDOM_SEED = 2021


def cluster_split(df: pd.DataFrame, train_ratio: float) -> Tuple[List[int], List[int]]:
    """Get train and test indices for a DataFrame with cluster labels.

    Similar to scikit-learn's stratified ``train_test_split``, but ensures that
    the train set with ratio of r_0 is a subset of the train set for ratio r_1,
    where r_1 > r_0.

    Args:
        df: A DataFrame containing a `cluster` column.

    Returns:
        train_indices
        test_indices

    """
    shuffled_df = df.sample(frac=1, random_state=RANDOM_SEED)
    classes = df["cluster"].unique()

    # Organise the data by which indexes belong to each class
    test_bins = {
        class_: list(shuffled_df.index[shuffled_df["cluster"] == class_])
        for class_ in classes
    }

    # Selectively remove the indexes until we get the correct ratios
    train_bins = defaultdict(list)
    for class_, indexes in test_bins.items():
        total_num_index = len(indexes)
        curr_train_ratio = lambda: len(train_bins[class_]) / total_num_index
        while curr_train_ratio() < train_ratio:
            train_bins[class_].append(indexes.pop())

    all_train_idxs = list(itertools.chain.from_iterable(train_bins.values()))
    all_test_idxs = list(itertools.chain.from_iterable(test_bins.values()))

    print(f"True train ratio = {len(all_train_idxs) / len(df.index)}")
    return all_train_idxs, all_test_idxs


def get_polyoxyethylene_smiles(num_head: int, num_tail: int) -> str:
    """Get a SMILES string for a polyoxyethylene alcohol."""
    head = "C" * num_head
    tail = "CCO" * num_tail
    return head + "O" + tail


class GraphData(Dataset):
    """Handle graph data subsets."""

    def __init__(self, graphs: List[Graph]):
        """Store graphs."""
        self.graphs = graphs
        super().__init__()

    def read(self) -> List[Graph]:
        """Return the graphs for this subset."""
        return self.graphs


class Datasets(Enum):
    """Available datasets."""

    QIN = DATASET_FOLDER / "qin-data.csv"
    NIST_NEW = DATASET_FOLDER / "nist-new-vals.csv"


def get_nist_data(
    mol_featuriser: MolNodeFeaturizer,
    preprocess: Optional[LayerPreprocess] = None,
) -> Tuple[DisjointLoader, pd.DataFrame]:
    """Get a data loader for the NIST anionics data."""
    df = pd.read_csv(Datasets.NIST_NEW.value, header=0)
    df["Molecules"] = [MolFromSmiles(smiles) for smiles in df["SMILES"]]

    # df["Convertable"] = ~df["SMILES"].str.contains(r"(Mn)|(Cs)|(Mg)")
    # convertable_df = df[df["Convertable"]]

    # graphs = mols_to_graph(list(convertable_df["Molecules"]), mol_featuriser, list(convertable_df["log CMC"]))
    graphs = mols_to_graph(list(df["Molecules"]), mol_featuriser, list(df["log CMC"]))
    graphs = list(map(preprocess, graphs)) if preprocess is not None else graphs

    return DisjointLoader(GraphData(graphs), shuffle=False), df


class QinDatasets(Enum):
    """Qin datasets split by test and train subsets."""

    QIN_NONIONICS_RESULTS = DATASET_FOLDER / "qin_nonionic_results.csv"
    QIN_ALL_RESULTS = DATASET_FOLDER / "qin_all_results.csv"


class DataReader:
    """Handle reading datasets from disk and preprocessing, plus cross-validation splitting."""

    def __init__(self, dataset: Datasets) -> None:
        """Read data from disk."""
        self.df = pd.read_csv(dataset.value, header=0)
        try:
            smiles = self.df["smiles"]
        except KeyError:
            smiles = self.df["SMILES"]
        self.df["Molecules"] = [MolFromSmiles(smiles) for smiles in smiles]

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


class DataLoader(ABC):
    """Handle reading Qin datasets from file."""

    def __init__(
        self,
        dataset: Union[QinDatasets, Datasets],
        num_splits: Optional[int] = None,
        num_repeats: Optional[int] = None,
        fold_idx: Optional[int] = None,
    ) -> None:
        """Load data and find train/test indexes.

        Args:
            dataset: Which Qin dataset to load.
            num_splits: The number of folds to split into.
            num_repeats: The number of repeats to perform.
            fold_idx: The fold to train.

        """
        self.df = pd.read_csv(dataset.value, header=0, index_col=0)
        smiles_col = (
            self.df["smiles"] if "smiles" in self.df.columns else self.df["SMILES"]
        )
        self.df["Molecules"] = [MolFromSmiles(smiles) for smiles in smiles_col]

        if num_splits is not None:
            # self.train_idxs, self.test_idxs = train_test_split(
            #     self.df.index,
            #     train_size=train_ratio,
            #     stratify=self.df["cluster"],
            #     random_state=RANDOM_SEED,
            # )
            rsk = RepeatedStratifiedKFold(
                n_splits=num_splits, n_repeats=num_repeats, random_state=RANDOM_SEED
            )
            all_train_idx, all_test_idx = list(zip(*rsk.split(self.df.index, self.df["cluster"])))
            self.train_idxs, self.test_idxs = (
                all_train_idx[fold_idx],
                all_test_idx[fold_idx],
            )
            self.optim_idxs, self.val_idxs = train_test_split(
                self.train_idxs, train_size=0.9, random_state=RANDOM_SEED
            )
        else:
            try:
                self.test_idxs = np.where(self.df["traintest"] == "test")[0]
                self.train_idxs = np.where(self.df["traintest"] == "train")[0]
                self.optim_idxs, self.val_idxs = train_test_split(
                    self.train_idxs, train_size=0.9, random_state=RANDOM_SEED
                )
            except (KeyError, ValueError):
                # No train/test split
                self.test_idxs = self.df.index
                self.train_idxs = np.array([])
                self.optim_idxs = self.train_idxs
                self.val_idxs = self.train_idxs


class ECFPData(DataLoader):
    """Handle reading Qin datasets from file and featurising with ECFP fingerprints."""

    def __init__(
        self,
        dataset: Union[Datasets, QinDatasets],
        hash_file: Optional[Path] = None,
        num_splits: Optional[int] = None,
        num_repeats: Optional[int] = None,
        fold_idx: Optional[int] = None,
    ) -> None:
        """Load data and initialise featuriser.

        Args:
            dataset: Which Qin dataset to load.
            hash_file: Where to save/load hash data to.

        """
        super().__init__(
            dataset, num_splits=num_splits, num_repeats=num_repeats, fold_idx=fold_idx
        )

        smiles_hashes = None
        save_hashes = False
        add_new_hashes = True

        if hash_file is not None:
            if hash_file.exists():
                smiles_hashes = SMILESHashes.load(hash_file)
                add_new_hashes = False
            else:
                save_hashes = True

        if smiles_hashes is None:
            smiles_hashes = SMILESHashes()

        self.smiles_hashes: SMILESHashes = smiles_hashes

        self.featuriser = ECFPCountFeaturiser(self.smiles_hashes)

        self.fingerprints = self.featuriser.featurise_molecules(
            list(self.df["Molecules"]), 2, add_new_hashes
        )
        if save_hashes:
            self.featuriser.smiles_hashes.save(hash_file)

    @property
    def expected(self) -> np.ndarray:
        """Get the expected (target) values as a numpy array."""
        try:
            return self.df.exp.to_numpy()
        except AttributeError:
            return self.df["log CMC"].to_numpy()

    def get_at_idxs(self, indexes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the fingerprints and target values for the given indexes."""
        return self.fingerprints[indexes, :], self.expected[indexes]

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
        return self.fingerprints, self.expected


class QinGraphData(DataLoader):
    """Handle reading Qin datasets from file and splitting into train and test subsets."""

    def __init__(
        self,
        dataset: QinDatasets,
        mol_featuriser: MolNodeFeaturizer = MolNodeFeaturizer(),
        preprocess: Optional[LayerPreprocess] = None,
        num_splits: Optional[int] = None,
        num_repeats: Optional[int] = None,
        fold_idx: Optional[int] = None,
    ) -> None:
        """Load data and initialise featuriser.

        Args:
            dataset: Which dataset to load.
            mol_featuriser: The molecular featuriser to use. This is important for consistency with featurising, e.g. one hot encoding.

        """
        super().__init__(
            dataset, num_splits=num_splits, num_repeats=num_repeats, fold_idx=fold_idx
        )

        self.mol_featuriser = mol_featuriser
        graphs = mols_to_graph(
            list(self.df["Molecules"]), self.mol_featuriser, list(self.df["exp"])
        )
        self.graphs = (
            list(map(preprocess, graphs)) if preprocess is not None else graphs
        )

    @property
    def train_dataset(self):
        """Get the training dataset."""
        train_graphs = [self.graphs[i] for i in self.train_idxs]
        return GraphData(train_graphs)

    @property
    def optim_dataset(self):
        """Get the optimisation dataset."""
        optim_graphs = [self.graphs[i] for i in self.optim_idxs]
        return GraphData(optim_graphs)

    @property
    def val_dataset(self):
        """Get the validation dataset."""
        val_graphs = [self.graphs[i] for i in self.val_idxs]
        return GraphData(val_graphs)

    @property
    def test_dataset(self):
        """Get the test dataset."""
        test_graphs = [self.graphs[i] for i in self.test_idxs]
        return GraphData(test_graphs)

    @property
    def all_dataset(self):
        """Get the full dataset."""
        return GraphData(self.graphs)

    @property
    def train_loader(self):
        """Get the training data loader."""
        return DisjointLoader(self.train_dataset)

    @property
    def optim_loader(self):
        """Get the optimisation data loader."""
        return DisjointLoader(self.optim_dataset)

    @property
    def train_loader_no_shuffle(self):
        """Get the training data loader."""
        return DisjointLoader(self.train_dataset, shuffle=False)

    @property
    def optim_loader_no_shuffle(self):
        """Get the optimisation data loader."""
        return DisjointLoader(self.optim_dataset, shuffle=False)

    @property
    def val_loader(self):
        """Get the validation data loader."""
        return DisjointLoader(self.val_dataset, shuffle=False)

    @property
    def test_loader(self):
        """Get the test data loader."""
        return DisjointLoader(self.test_dataset, shuffle=False)

    @property
    def all_loader(self):
        """Get the full data loader."""
        return DisjointLoader(self.all_dataset, shuffle=False)


def get_nist_and_qin(
    mol_featuriser: MolNodeFeaturizer,
    preprocess: Optional[LayerPreprocess] = None,
) -> Tuple[DisjointLoader, pd.DataFrame]:
    """Get a loader for all of the NIST and Qin data, for visualisation."""
    nist_df = pd.read_csv(Datasets.NIST_NEW.value, header=0)
    nist_df["Molecules"] = [MolFromSmiles(smiles) for smiles in nist_df["SMILES"]]

    # These elements don't appear in the Qin dataset, so they don't have a one-hot encoding.
    nist_df["Convertable"] = ~nist_df["SMILES"].str.contains(r"(Mn)|(Cs)|(Mg)")
    convertable_df = nist_df[nist_df["Convertable"]]

    nist_graphs = mols_to_graph(
        list(convertable_df["Molecules"]),
        mol_featuriser,
        list(convertable_df["log CMC"]),
    )
    nist_graphs = mols_to_graph(
        list(nist_df["Molecules"]), mol_featuriser, list(nist_df["log CMC"])
    )
    nist_graphs = (
        list(map(preprocess, nist_graphs)) if preprocess is not None else nist_graphs
    )

    qin_loader = QinGraphData(QinDatasets.QIN_ALL_RESULTS, mol_featuriser, preprocess)
    qin_df = qin_loader.df
    qin_df.rename(columns={"smiles": "SMILES", "exp": "log CMC"}, inplace=True)
    qin_graphs = qin_loader.all_dataset.read()

    combined_df = pd.concat([qin_df, convertable_df], ignore_index=True)
    combined_graphs = qin_graphs + nist_graphs

    return DisjointLoader(GraphData(combined_graphs), shuffle=False), combined_df
