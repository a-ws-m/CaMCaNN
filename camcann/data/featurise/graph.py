"""Utilities for featurising graph nodes and edges."""
from operator import methodcaller
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit.Chem.rdchem import Atom, Mol
from sklearn.preprocessing import OneHotEncoder

MolecularNodeMatrix = np.ndarray
AllNodesMatrix = np.ndarray

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

"""A list of atom properties to convert to one-hot vectors by default."""
DEFAULT_ONE_HOT_SPEC: List[str] = [
    "number",
    "degree",
    "implicit_valence",
    "hybridization",
    "total_num_H",
]


class NotTrainedError(Exception):
    def __init__(self, message: Optional[str] = None, *args: object) -> None:
        if message is None:
            message = (
                "No trained one-hot parameters, "
                "first call `.one_hot_encode(..., train=True)`."
            )
        super().__init__(message, *args)


def array_to_mol_list(
    array: AllNodesMatrix, molecules: List[Mol]
) -> List[MolecularNodeMatrix]:
    """Split a concatenated matrix to a list of matrices.
    
    The input matrix should have shape (total_num_atoms, num_features).
    `molecules` should be a list of vectors representing the atomic
    numbers of every atom in a molecule. The order of the atoms
    in `array` must therefore match the order of the `molecules`
    and their constituent atoms.
    
    """
    loc: int = 0
    mat_list: List[MolecularNodeMatrix] = []
    for molecule in molecules:
        num_atoms: int = molecule.GetNumAtoms()
        mat_list.append(array[loc : loc + num_atoms])
        loc += num_atoms

    return mat_list


class MolNodeFeaturizer:
    """Featurize atoms in molecular graphs."""

    def __init__(
        self, sparse_encoding: bool = False, encoder_params: Optional[dict] = None
    ) -> None:
        self.one_hot = OneHotEncoder(sparse=sparse_encoding)
        self.one_hot_trained: bool = encoder_params is not None
        if self.one_hot_trained:
            self.one_hot.set_params(**encoder_params)

    def featurize_all_molecules(
        self, molecules: List[Mol],
    ) -> List[MolecularNodeMatrix]:
        """Convert a list of rdkit molecules to a list of molecular graph node matrices."""
        # A list of the molecular matrices for just categorical atomic features
        mol_cat_matrices: List[MolecularNodeMatrix] = []
        # A list of the molecular matrices for just the numerical atomic features
        mol_num_matrices: List[MolecularNodeMatrix] = []
        for mol in molecules:
            atoms: List[Atom] = list(mol.GetAtoms())
            atom_cat_feats, atom_num_feats = self.featurize_atoms(atoms)
            mol_cat_matrices.append(atom_cat_feats)
            mol_num_matrices.append(atom_num_feats)

        # * Convert the categorical matrices
        all_mol_cat_matrix = self.one_hot_encode(
            mol_cat_matrices, train=not self.one_hot_trained
        )
        # * Stack all the numerical matrices
        all_mol_num_matrix = np.vstack(mol_num_matrices)
        # * Combine the two horizontally
        all_mol_atom_matrix = np.hstack([all_mol_cat_matrix, all_mol_num_matrix])
        # * Subdivide the array row-wise into a list of molecular node matrices
        return array_to_mol_list(all_mol_atom_matrix, molecules)

    def featurize_atoms(
        self, atoms: List[Atom]
    ) -> Tuple[MolecularNodeMatrix, MolecularNodeMatrix]:
        """Featurize a list of atoms."""
        atom_cat_feats: List[List[int]] = []
        atom_num_feats: List[List[int]] = []
        for atom in atoms:
            cat_feats = [
                RDCHEM_ATOM_METHS[cat_method](atom)
                for cat_method in DEFAULT_ONE_HOT_SPEC
            ]
            num_feats = [
                method(atom)
                for key, method in RDCHEM_ATOM_METHS.items()
                if key not in DEFAULT_ONE_HOT_SPEC
            ]
            atom_cat_feats.append(cat_feats)
            atom_num_feats.append(num_feats)
        return np.array(atom_cat_feats), np.array(atom_num_feats)

    def one_hot_encode(
        self, node_cats: List[MolecularNodeMatrix], train: bool = False,
    ) -> AllNodesMatrix:
        """Get one-hot encodings for node categories."""
        all_atom_matrix = np.vstack(node_cats)
        if train:
            self.one_hot_trained = True
            return self.one_hot.fit_transform(all_atom_matrix)
        elif not self.one_hot_trained:
            raise NotTrainedError()
        else:
            return self.one_hot.transform(all_atom_matrix)

    @property
    def one_hot_params(self) -> dict:
        """Get the current parameters of the one-hot encoder."""
        if not self.one_hot_trained:
            raise NotTrainedError()
        return self.one_hot.get_params()
