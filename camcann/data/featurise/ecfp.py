"""Molecular featurisation utilities."""
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from operator import getitem, methodcaller
from os import PathLike
from typing import (Callable, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Set,
                    Tuple, Union)
from zlib import crc32

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolFragmentToSmiles
from rdkit.Chem.rdchem import Atom, Bond, Mol
from scipy.sparse import csr_array

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


def hash_array(arr: np.ndarray) -> float:
    """Compute the CRC32 hash of an array of numbers."""
    return crc32(arr.tobytes())


class BondType(IntEnum):
    """Bond types."""

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    IONIC = 5


class SimpleBond(NamedTuple):
    """Contain minimalistic information about bonds."""

    atom_idxs: Tuple[int, int]
    type_: BondType

    @classmethod
    def from_rdk(cls: "SimpleBond", bond: Bond) -> "SimpleBond":
        """Extract information from an RDKit ``Bond``."""
        return cls(
            atom_idxs=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            type_=getitem(BondType, str(bond.GetBondType())),
        )

class SMILESHashes:
    """Contains the fingerprint index and SMILES string associated with each hash."""
    
    def __init__(self, hash_df: Optional[pd.DataFrame]=None) -> None:
        """Initialize hash DataFrame."""
        if hash_df is None:
            hash_df = pd.DataFrame({"fingerprint_index": [], "SMILES": []})

        self.hash_df = hash_df
    
    def setdefault(self, hash_: int, smiles: str) -> int:
        """Get the index for a given hash, generating a new one if necessary."""
        try:
            return self.hash_df.loc[hash_].fingerprint_index
        except KeyError:
            new_index = len(self)
            self.hash_df.loc[hash_] = {"fingerprint_index": new_index, "SMILES": smiles}
            return new_index
    
    def get_hash_idx(self, hash_: int) -> Optional[int]:
        """Get the fingerprint index of a hash, if it is in the DataFrame."""
        try:
            return self.hash_df.loc[hash_].fingerprint_index
        except KeyError:
            return None
    
    def __len__(self) -> int:
        """Get the number of hash entries."""
        return len(self.hash_df.index)

    @property
    def smiles(self) -> List[str]:
        """Get the SMILES strings in the DataFrame."""
        return self.hash_df.SMILES.tolist()
    
    def save(self, path: PathLike):
        """Save hash dataframe to path."""
        self.hash_df.to_csv(path)
    
    @classmethod
    def load(cls, path: PathLike):
        """Load from a hash dataframe."""
        return cls(pd.read_csv(path, header=0, index_col=0))

def frag_to_smiles(mol: Mol, atoms: Union[int, Sequence[int]]) -> str:
    """Get the SMILES string for a given fragment."""
    try:
        len(atoms)
    except TypeError:
        atoms = [atoms]
    return MolFragmentToSmiles(mol, atomsToUse=list(atoms), allHsExplicit=True)

@dataclass
class HashedMolecule:
    """Contains atomic hashes and atomic bonds."""

    atom_hashes: List[int]
    bonds: List[SimpleBond]
    rdk_mol: Mol

    def __post_init__(self) -> None:
        """Compute the atom interaction matrix and initialize atomic walks.

        The atom interaction matrix is a list defining which atom indexes to
        look at, and in what order, when computing a step update. An
        interaction, in this case, is a hash function computed over a list of
        bonded atoms' hashes.

        The atomic walk dictionary indicates the set of atom indexes that are
        visited during a walk of a given length.

        """
        bonded_atoms: List[List[Tuple[int, BondType]]] = [[] for _ in range(len(self.atom_hashes))]
        for bond in self.bonds:
            for pair_idx in (0, 1):
                # Add reference to other atom for both atoms in the bond
                bond_tuple = (bond.atom_idxs[pair_idx], bond.type_)
                bonded_atoms[bond.atom_idxs[1 - pair_idx]].append(bond_tuple)

        # Sort atomic interactions based primarily on the type of bond and
        # secondarily on the atomic hash magnitude.
        self.atom_interactions: List[List[int]] = [
            [bonded_atom[0] for bonded_atom in sorted(entry, key=lambda bond_info: (bond_info[1], self.atom_hashes[bond_info[0]]))]
            for entry in bonded_atoms
        ]
        for idx, interactions in enumerate(self.atom_interactions):
            # Add self-interactions
            interactions.insert(0, idx)
        
        # Initialize atomic walk dictionary
        self.atomic_walks: Dict[int, List[FrozenSet[int]]] = {0: [frozenset({idx}) for idx in range(len(self.atom_hashes))]}
        # Initialize cumulative atomic walks dictionary
        self.cum_atomic_walks: Dict[int, List[FrozenSet[int]]] = {1: self.atomic_walks[0]}

    @classmethod
    def from_rdk(
        cls: "HashedMolecule", molecule: Mol, hashes: List[int]
    ) -> "HashedMolecule":
        """Make a HashedMolecule from an RDKit ``Molecule`` and a list of hashes."""
        return cls(
            atom_hashes=hashes,
            bonds=[SimpleBond.from_rdk(bond) for bond in molecule.GetBonds()],
            rdk_mol=molecule,
        )

    def hash_step(self):
        """Compute a single hash update step using bonded hashes."""
        new_hashes = []
        for interactions in self.atom_interactions:
            new_hashes.append(hash_array(np.array([self.atom_hashes[idx] for idx in interactions])))
        
        self.atom_hashes = new_hashes
    
    def _update_atomic_walks(self):
        """Add another step to the :attr:`atomic_walks`.
        
        This is used when keeping track of duplicate substructures.
        
        """
        # Find largest step that we've already computed
        largest_step = max(self.atomic_walks.keys())

        # Do another step
        new_walks = []
        for walk in self.atomic_walks[largest_step]:
            walk_buffer = set()
            for atom_idx in walk:
                walk_buffer.update(self.atom_interactions[atom_idx])

            new_walk = walk | walk_buffer
            new_walks.append(new_walk)

        self.atomic_walks[largest_step + 1] = new_walks
        self.cum_atomic_walks[largest_step + 2] =  [cum_walk | set(new_walk) for cum_walk, new_walk in zip(self.cum_atomic_walks[largest_step + 1], new_walks)]

    def check_duplicates(self, radius: int) -> List[Tuple[int, Optional[int]]]:
        """Get a list of hash indexes that refer to same substructure at a given radius.
        
        Update atomic walks dictionary along the way.

        Returns:
            A list of `(index_1, index_2)`. Sometimes, substructures may be
            duplicates of a hash from a previous iteration. In this case, a
            tuple of `(index, None)` is returned.
        
        """
        # Generate atom index sets for a walk of a given length
        while radius not in self.atomic_walks:
            self._update_atomic_walks()
        current_lvl_walks = self.atomic_walks[radius]
        cumulative_walks = self.cum_atomic_walks[radius]

        duplicates = []
        for idx, walk in enumerate(current_lvl_walks):
            if walk in cumulative_walks:
                duplicates.append((idx, None))
                continue
            try:
                duplicates.append((idx, current_lvl_walks.index(walk, idx+1)))
            except ValueError:
                # Set doesn't appear more than once in this level
                continue

        return duplicates
    
    def get_hash_list(self, num_steps: int, smiles_hashes: Optional[SMILESHashes]=None) -> List[int]:
        """Get a list of all hashes generated during a given number of steps.
        
        Args:
            num_steps: The total number of atomic steps to perform.
            smiles_hashes: An optional :class:`SMILESHashes` instance to update
                with SMILES fragments associated with each hash.
        
        """
        hashes = self.atom_hashes[:]
        include_smiles = smiles_hashes is not None
        if include_smiles:
            while 2 not in self.atomic_walks:
                # Need to have these calculated for figuring out SMILES fragments.
                self._update_atomic_walks()
            smiles = [frag_to_smiles(self.rdk_mol, cum_walk) for cum_walk in self.cum_atomic_walks[1]]

        for steps_done in range(num_steps):
            self.hash_step()
            new_hashes = self.atom_hashes[:]
            if include_smiles:
                new_smiles = [frag_to_smiles(self.rdk_mol, cum_walk) for cum_walk in self.cum_atomic_walks[2 + steps_done]]

            if steps_done > 0:
                # Potentially duplicate hashes in list
                duplicates = self.check_duplicates(steps_done + 1)

                to_delete = []
                for indexes in duplicates:
                    if indexes[1] is None:
                        to_delete.append(indexes[0])
                    else:
                        lower_hash_idx = indexes[new_hashes[indexes[1]] < new_hashes[indexes[0]]]
                        to_delete.append(lower_hash_idx)

                # Delete values
                for delete_idx in sorted(to_delete, reverse=True):
                    new_hashes.pop(delete_idx)
                    if include_smiles:
                        new_smiles.pop(delete_idx)
            
            hashes.extend(new_hashes)
            if include_smiles:
                smiles.extend(new_smiles)
        
        if include_smiles:
            for hash_, smile in zip(hashes, smiles):
                smiles_hashes.setdefault(hash_, smile)
        return hashes


class ECFPCountFeaturiser:
    """Convert molecules to ECFPs with a canonical count of each circular group."""

    def __init__(self, smiles_hashes: SMILESHashes = SMILESHashes()) -> None:
        """Initialize SMILES hashes."""
        self.smiles_hashes = smiles_hashes

    def featurise_atoms(self, molecule: Mol) -> List[int]:
        """Get the hashed features of a molecule's atoms."""
        atom_hashes = []
        for atom in molecule.GetAtoms():
            atom_features = np.array([method(atom) for method in RDCHEM_ATOM_METHS.values()])
            atom_hash = hash_array(atom_features)
            atom_hashes.append(atom_hash)

        return atom_hashes

    def _initial_hash(self, molecules: Mol) -> List[HashedMolecule]:
        """Get the initial hashes for atoms in a list of molecules.

        Args:
            molecules: The molecules to hash.

        Returns:
            A list of the molecules' hash lists, each of which containing
            the atoms' hashes in the RDKit order.

        """
        mols_atom_hashes = [self.featurise_atoms(mol) for mol in molecules]
        return [
            HashedMolecule.from_rdk(mol, atom_hashes)
            for mol, atom_hashes in zip(molecules, mols_atom_hashes)
        ]

    def featurise_molecules(self, molecules: List[Mol], radius: int, add_new_hashes: bool = True) -> np.ndarray:
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
        hashed_molecules = self._initial_hash(molecules)

        get_hash_args = (radius, self.smiles_hashes) if add_new_hashes else (radius,)
        hash_lists = [mol.get_hash_list(*get_hash_args) for mol in hashed_molecules]

        hash_counters = [Counter(hash_list) for hash_list in hash_lists]

        # Convert hashes to indexes and prepare csr_array; see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html
        counts = []
        idxptr = [0]
        hash_idxs = []
        for hash_counter in hash_counters:
            for hash_, count in hash_counter.items():
                hash_idx = self.smiles_hashes.get_hash_idx(hash_)
                if hash_idx is None:
                    continue

                counts.append(count)
                hash_idxs.append(hash_idx)
            idxptr.append(len(hash_idxs))

        count_array = csr_array(
            (counts, hash_idxs, idxptr)
        ).toarray()

        # Pad with extra zeroes as needed
        return self.pad_count_array(count_array)
    
    def pad_count_array(self, count_array: np.ndarray) -> np.ndarray:
        """Pad a count array with zeroes till its features match the number of :attr:`smiles_hashes`."""
        num_padding_cols = len(self.smiles_hashes) - count_array.shape[1]
        if num_padding_cols:
            return np.pad(count_array, (0, num_padding_cols), "constant")
        return count_array
    
    def label_features(self, count_array: np.ndarray, original_smiles: Optional[List[str]]=None) -> pd.DataFrame:
        """Label the features of a count array based on the current :attr:`smiles_hashes`."""
        # Pad with extra zeroes as needed
        column_labels = [f"Num {smiles}" for smiles in self.smiles_hashes.smiles]
        return pd.DataFrame(self.pad_count_array(count_array), columns=column_labels, index=original_smiles)

if __name__ == "__main__":
    # Quick and dirty test
    test_smiles = ["C(C)(C)CC"]
    test_molecules = [MolFromSmiles(test_smile) for test_smile in test_smiles]

    featuriser = ECFPCountFeaturiser()
    fingerprints = featuriser.featurise_molecules(test_molecules, 2)

    print(featuriser.label_features(fingerprints, test_smiles))
