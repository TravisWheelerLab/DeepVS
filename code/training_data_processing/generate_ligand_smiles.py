from copy import deepcopy
import os
import pickle
import sys
import numpy as np
import code.utils.data_processing_utils as data_utils
import code.utils.pdb_utils as pdb_utils
import code.utils.pocket_gen_utils as pocket_gen_utils
from torch_geometric.nn import radius_graph 
from torch_geometric.nn import knn
from torch_geometric.utils import add_self_loops, to_undirected

import rdkit.Chem as Chem
from rdkit.Chem import AllChem


# from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords
# from code.utils.get_distance import get_distance

import torch
from torch_geometric.data import Data

def generate_ligand_smiles(
    id_batch: list,
    pdbbind_dir: str,
    mol_graph_dir: str = None,
    data_dir: str = None,
    **kwargs
) -> None:
    mol_graph_dir, mol_graph_ft = data_utils.get_output_paths(
        mol_graph_dir, data_dir, kwargs["mol_graph_file_template"]
    )

    mol2_ft = pdbbind_dir + '%s/%s_ligand.mol2'

    POCKET_ATOM_LABELS = kwargs['POCKET_ATOM_LABELS']
    INTERACTION_LABELS = kwargs['INTERACTION_LABELS']

    for target_index, pdb_id in enumerate(id_batch):
        print(pdb_id)
        mol2_mol = Chem.MolFromMol2File(mol2_ft % (pdb_id, pdb_id))
        smiles_string = Chem.MolToSmiles(mol2_mol, isomericSmiles=True)
        print(smiles_string)