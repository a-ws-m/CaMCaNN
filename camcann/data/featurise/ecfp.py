"""Molecular featurisation utilities."""
from collections import Counter, defaultdict
from itertools import chain
from operator import methodcaller
from os import PathLike
from pathlib import Path
from typing import (
    Callable,
    DefaultDict,
    Dict,
    FrozenSet,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    TypeAlias,
    Tuple,
    Union,
)
from zlib import crc32

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolFragmentToSmiles
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN
from rdkit.Chem.rdchem import Atom, Bond, Mol
from scipy.sparse import csr_array

AtomicIndex: TypeAlias = int
BondIndex: TypeAlias = int
AtomicHash: TypeAlias = int
AtomicEnvHash: TypeAlias = int

RDCHEM_ATOM_METH_NAMES: Dict[str, str] = {
    "number": "GetAtomicNum",
    "degree": "GetDegree",
    "implicit_valence": "GetImplicitValence",
    "formal_charge": "GetFormalCharge",
    "num_radical_electrons": "GetNumRadicalElectrons",
    "hybridization": "GetHybridization",
    "is_aromatic": "GetIsAromatic",
    "total_num_H": "GetTotalNumHs",
}

"""A dictionary of atomic properties and a callable to extract them from an `Atom`."""
RDCHEM_ATOM_METHS: Dict[str, Callable[[Atom], Union[str, float, int]]] = {
    key: methodcaller(value) for key, value in RDCHEM_ATOM_METH_NAMES.items()
}


def hash_array(arr: np.ndarray) -> AtomicHash:
    """Compute the CRC32 hash of an array of numbers."""
    return crc32(arr.tobytes())


def get_atom_hash(atom: Atom) -> AtomicHash:
    """Extract information from an RDKit ``Atom``."""
    atom_feats = np.array(
        list(atom_meth(atom) for atom_meth in RDCHEM_ATOM_METHS.values())
    )
    return hash_array(atom_feats)


def get_bonds_atoms(bond: Bond) -> Tuple[AtomicIndex, AtomicIndex]:
    """Get the atomic indexes that a bond connects."""
    return (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())


def frag_to_smiles(mol: Mol, atoms: Union[int, Iterable[int]]) -> str:
    """Get the SMILES string for a given fragment."""
    if isinstance(atoms, int):
        atoms = [atoms]
    return MolFragmentToSmiles(mol, atomsToUse=list(atoms), allHsExplicit=True)


class SimpleMolecule(NamedTuple):
    """Contain minimalistic information about a molecule."""

    atoms: Dict[AtomicIndex, AtomicHash]
    bonds: Dict[BondIndex, Tuple[AtomicIndex, AtomicIndex]]
    atom_envs: Dict[int, Set[FrozenSet[BondIndex]]]
    atom_env_counts: Dict[AtomicEnvHash, int]
    atom_env_smiles: Dict[AtomicEnvHash, str]

    @classmethod
    def from_rdk(cls, mol: Mol, max_radius: int) -> "SimpleMolecule":
        """Extract information from an RDKit ``Mol``."""
        rdk_atoms = list(mol.GetAtoms())
        rdk_bonds = list(mol.GetBonds())

        atoms = {idx: get_atom_hash(atom) for idx, atom in enumerate(rdk_atoms)}
        bonds = {idx: get_bonds_atoms(bond) for idx, bond in enumerate(rdk_bonds)}

        atom_envs: Dict[int, Set[FrozenSet[BondIndex]]] = dict()
        for radius in range(1, max_radius + 1):
            atom_env_buff: Set[FrozenSet[BondIndex]] = set()
            for atom_idx in atoms.keys():
                env = frozenset(FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx))
                if len(env):
                    atom_env_buff.add(env)

            atom_envs[radius] = atom_env_buff

        atom_env_count_buff: DefaultDict[AtomicEnvHash, int] = defaultdict(int)
        atom_env_smiles: Dict[AtomicEnvHash, str] = dict()
        # * First, radius zero
        for atom_idx, hash_ in atoms.items():
            atom_env_count_buff[hash_] += 1
            atom_env_smiles[hash_] = frag_to_smiles(mol, atom_idx)

        # * Then all the rest!
        all_bond_combs: Set[FrozenSet[BondIndex]] = set()
        for atom_env_set in atom_envs.values():
            all_bond_combs.update(atom_env_set)

        for bond_comb in all_bond_combs:
            atom_idxs: Set[AtomicIndex] = set(
                chain.from_iterable(bonds[bond_idx] for bond_idx in bond_comb)
            )

            atom_env: List[AtomicHash] = sorted(
                atoms[atom_idx] for atom_idx in atom_idxs
            )
            atom_env_hash: AtomicEnvHash = hash_array(np.array(atom_env))

            atom_env_count_buff[atom_env_hash] += 1
            atom_env_smiles[atom_env_hash] = frag_to_smiles(mol, atom_idxs)

        return cls(atoms, bonds, atom_envs, dict(atom_env_count_buff), atom_env_smiles)

    @classmethod
    def from_smiles(cls, smiles: str, max_radius: int) -> "SimpleMolecule":
        """Extract information from a SMILES string."""
        return cls.from_rdk(MolFromSmiles(smiles), max_radius)


class SMILESHashes:
    """Contains information about each subgraph.

    This information includes:
        * The hash (an index)
        * The SMILES string for that hash (not unique)
        * An optional weight associated with the subgraph for a linear model.
        * A boolean value stating whether the subgraph has been feature selected.

    """

    def __init__(self, hash_df: Optional[pd.DataFrame] = None) -> None:
        """Initialize hash DataFrame."""
        if hash_df is None:
            hash_df = pd.DataFrame(
                {
                    "fingerprint_index": [],
                    "SMILES": [],
                    "weight": [],
                    "selected": [],
                    "above_threshold_occurrence": [],
                    "norm_weight": [],
                }
            )

        self.hash_df = hash_df

    def setdefault(self, hash_: int, smiles: str) -> int:
        """Get the index for a given hash, generating a new one if necessary."""
        try:
            return self.hash_df.loc[hash_].fingerprint_index
        except KeyError:
            new_index = len(self)
            self.hash_df.loc[hash_] = {
                "fingerprint_index": new_index,
                "SMILES": smiles,
                "weight": pd.NA,
                "selected": pd.NA,
                "above_threshold_occurrence": pd.NA,
                "norm_weight": pd.NA,
            }
            return new_index

    def get_hash_idx(self, hash_: int) -> Optional[int]:
        """Get the fingerprint index of a hash, if it is in the DataFrame."""
        try:
            return self.hash_df.loc[hash_].fingerprint_index
        except KeyError:
            return None

    def set_regularised_selection(self, support: np.ndarray):
        """Set the ``selected`` column of the data frame based on a support array.

        The support array can be acquired using a scikit-learn
        ``SelectFromModel`` selector. This must be used after the initial
        removal of features below the threshold occurance. This function
        determines the binary ``AND`` of ``above_threshold_occurrence`` and
        ``support``, filling in the missing values.

        """
        if self.hash_df["above_threshold_occurrence"].isnull().any():
            raise ValueError(
                "One or more subgraphs have not been checked for being above threshold occurrence."
            )

        selected = self.hash_df["above_threshold_occurrence"].to_numpy(copy=True)
        selected[selected] = support
        self.hash_df["selected"] = selected

    def set_weights(self, norm_weights: np.ndarray, weights: np.ndarray):
        """Set the weights of the has dataframe."""
        selected = self.hash_df["selected"].values
        try:
            self.hash_df.loc[selected, "weight"] = weights.flatten()
        except ValueError as e:
            weights_size = weights.flatten().size
            selected_size = self.hash_df.loc[selected, "weight"].values.size

            print("Size mismatch when saving weights!")
            print(f"Size of included features: {selected_size}")
            print(f"Size of assigned weights: {weights_size}")

            raise e

        self.hash_df.loc[selected, "norm_weight"] = norm_weights.flatten()

    def __len__(self) -> int:
        """Get the number of hash entries."""
        return len(self.hash_df.index)

    @property
    def smiles(self) -> List[str]:
        """Get the SMILES strings in the DataFrame."""
        return self.hash_df.SMILES.tolist()

    @property
    def selected_idxs(self) -> np.ndarray:
        """Get the indexes of the feature selected subgraphs."""
        return np.nonzero(self.hash_df.selected)[0]

    def save(self, path: PathLike):
        """Save hash dataframe to path."""
        self.hash_df.to_csv(path)

    @classmethod
    def load(cls, path: PathLike):
        """Load from a hash dataframe."""
        return cls(pd.read_csv(path, header=0, index_col=0))


class ECFPCountFeaturiser:
    """Convert molecules to ECFPs with a canonical count of each circular group."""

    def __init__(self, smiles_hashes: SMILESHashes = SMILESHashes()) -> None:
        """Initialize SMILES hashes."""
        self.smiles_hashes = smiles_hashes

    def featurise_atoms(self, molecule: Mol) -> List[int]:
        """Get the hashed features of a molecule's atoms."""
        atom_hashes = []
        for atom in molecule.GetAtoms():
            atom_features = np.array(
                [method(atom) for method in RDCHEM_ATOM_METHS.values()]
            )
            atom_hash = hash_array(atom_features)
            atom_hashes.append(atom_hash)

        return atom_hashes

    def featurise_molecules(
        self, molecules: List[Mol], radius: int, add_new_hashes: bool = True
    ) -> np.ndarray:
        """Featurise molecules as count vectors.

        The :attr:`smiles_hashes` will be updated along the way. This means that
        multiple calls to this function may result in larger feature vectors. It
        is sufficient to pad the shorter feature vectors with zeroes, as by
        necessity, the remaining groups will have values of zero; this can
        be achieved using :meth:`pad_count_array`.

        Args:
            molecules: The molecules to featurise.
            radius: The number of adjacent atom groups to consider.
            add_new_hashes: Whether to add new hashes to the
                :attr:`smiles_hashes`. If ``False``, ignore newly encountered hashes.

        """
        simple_mols: List[SimpleMolecule] = [
            SimpleMolecule.from_rdk(mol, radius) for mol in molecules
        ]

        # Convert hashes to indexes and prepare csr_array; see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html
        counts = []
        idxptr = [0]
        hash_idxs = []
        for simple_mol in simple_mols:
            for hash_, count in simple_mol.atom_env_counts.items():
                hash_idx = self.smiles_hashes.get_hash_idx(hash_)
                if hash_idx is None:
                    if add_new_hashes:
                        hash_idx = self.smiles_hashes.setdefault(
                            hash_, simple_mol.atom_env_smiles[hash_]
                        )
                    else:
                        continue

                counts.append(count)
                hash_idxs.append(hash_idx)
            idxptr.append(len(hash_idxs))

        count_array = csr_array((counts, hash_idxs, idxptr)).toarray()

        # Pad with extra zeroes as needed
        return self.pad_count_array(count_array)

    def pad_count_array(self, count_array: np.ndarray) -> np.ndarray:
        """Pad a count array with zeroes till its features match the number of :attr:`smiles_hashes`."""
        num_padding_cols = len(self.smiles_hashes) - count_array.shape[1]
        if num_padding_cols:
            return np.pad(count_array, ((0, 0), (0, num_padding_cols)), "constant")
        return count_array

    def label_features(
        self, count_array: np.ndarray, original_smiles: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Label the features of a count array based on the current :attr:`smiles_hashes`."""
        # Pad with extra zeroes as needed
        column_labels = [f"Num {smiles}" for smiles in self.smiles_hashes.smiles]
        return pd.DataFrame(
            self.pad_count_array(count_array),
            columns=column_labels,
            index=original_smiles,
        )


if __name__ == "__main__":
    # Quick and dirty test
    all_results = pd.read_csv(
        Path(__file__).parents[1] / "datasets" / "qin_all_results.csv", header=0
    )
    test_smiles = all_results["smiles"]
    test_molecules = [MolFromSmiles(test_smile) for test_smile in test_smiles]

    featuriser = ECFPCountFeaturiser()
    fingerprints = featuriser.featurise_molecules(test_molecules, 2)

    smiles_series = featuriser.smiles_hashes.hash_df.SMILES
    non_unique = smiles_series.value_counts() > 1
    # We'd really like to make sure these are distinguished
    print(non_unique[non_unique])
