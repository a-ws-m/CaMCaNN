"""Molecular featurisation utilities."""
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from operator import itemgetter, methodcaller, getitem
from typing import Callable, Dict, List, NamedTuple, Tuple, Union
from zlib import crc32

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
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


@dataclass
class HashedMolecule:
    """Contains atomic hashes and atomic bonds."""

    atom_hashes: List[int]
    bonds: List[SimpleBond]

    def __post_init__(self) -> None:
        """Compute the atom interaction matrix.

        This is a list defining which atom indexes to look at, and in what
        order, when computing a step update. An interaction, in this case, is a
        hash function computed over a list of bonded atoms' hashes.

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

    @classmethod
    def from_rdk(
        cls: "HashedMolecule", molecule: Mol, hashes: List[int]
    ) -> "HashedMolecule":
        """Make a HashedMolecule from an RDKit ``Molecule`` and a list of hashes."""
        return cls(
            atom_hashes=hashes,
            bonds=[SimpleBond.from_rdk(bond) for bond in molecule.GetBonds()],
        )

    def hash_step(self):
        """Compute a single hash update step using bonded hashes."""
        for idx, interactions in enumerate(self.atom_interactions):
            self.atom_hashes[idx] = hash_array(np.array([self.atom_hashes[idx] for idx in interactions]))
        
        # * TODO: Remove duplicate substructure identifiers.
        # Achieve this by checking for duplicate substructures and discarding by setting atom_hash to a negative value.
        # Update the interaction matrix to remove bonds with the discarded identifier centre.


class ECFPCountFeaturiser:
    """Convert molecules to ECFPs with a canonical count of each circular group."""

    def __init__(self, vocabulary: Dict[int, int] = {}) -> None:
        """Initialize featuriser and its hashmap."""
        self.vocabulary = vocabulary

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

    def featurise_molecules(self, molecules: Mol, radius: int) -> np.ndarray:
        """Featurise molecules as count vectors.

        The :attr:`vocabulary` will be updated along the way. This means that
        multiple calls to this function may result in larger feature vectors. It
        is sufficient to pad the shorter feature vectors with zeroes, as by
        necessity, the remaining groups will have values of zero.

        Args:
            molecules: The molecules to featurise.
            radius: The number of adjacent atom groups to consider.

        """
        hashed_molecules = self._initial_hash(molecules)
        hash_counters = [
            Counter(hashed_mol.atom_hashes) for hashed_mol in hashed_molecules
        ]
        for _ in range(radius):
            for hash_counter, hashed_molecule in zip(hash_counters, hashed_molecules):
                hashed_molecule.hash_step()
                hash_counter.update(hashed_molecule.atom_hashes)

        # Convert hashes to indexes and prepare csr_array; see
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html
        counts = []
        idxptr = [0]
        hash_idxs = []
        for hash_counter in hash_counters:
            for hash_, count in hash_counter.items():
                hash_idx = self.vocabulary.setdefault(hash_, len(self.vocabulary))
                counts.append(count)
                hash_idxs.append(hash_idx)
            idxptr.append(len(hash_idxs))

        count_array = csr_array(
            (counts, hash_idxs, idxptr)
        ).toarray()

        # Pad with extra zeroes as needed
        num_padding_cols = len(self.vocabulary) - count_array.shape[1]
        if num_padding_cols:
            count_array = np.pad(count_array, (0, num_padding_cols), "constant")
        
        return count_array

if __name__ == "__main__":
    # Quick and dirty test
    test_smiles = ["CCC",  "CCCCC", "CC(=O)C", "CC(=O)CO"]
    test_molecules = [MolFromSmiles(test_smile) for test_smile in test_smiles]

    featuriser = ECFPCountFeaturiser()
    fingerprints = featuriser.featurise_molecules(test_molecules, 1)

    print(pd.DataFrame({"SMILES": test_smiles, "Fingerprint": [np.array2string(fp) for fp in fingerprints]}))
