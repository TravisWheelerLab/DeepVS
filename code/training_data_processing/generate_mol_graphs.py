import pickle
import os
import re
import numpy as np
import sys
import glob
import yaml
from copy import deepcopy
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import code.utils.data_processing_utils as data_utils
import code.utils.mol_gen_utils as mol_gen_utils


def generate_mol_graphs(
    config: dict,
    id_batch: list,
    pdbbind_dir: str,
    mol_graph_dir: str = None,
    data_dir: str = None,
    ip_dir: str = None,
    **kwargs
) -> None:
    mol_graph_dir, mol_graph_ft = data_utils.get_output_paths(
        mol_graph_dir, data_dir, config["mol_graph_file_template"]
    )
    ip_dir, ip_ft = data_utils.get_output_paths(
        ip_dir, data_dir, config["interaction_profile_file_template"]
    )

    if os.path.exists(mol_graph_dir) == False:
        os.makedirs(mol_graph_dir)

    INTERACTION_LABELS = config["INTERACTION_LABELS"]
    ATOM_LABELS = config["MOL_ATOM_LABELS"]
    # ['halogenbond', 'hbond_a', 'hbond_d', 'hydroph_interaction', 'pication_c', 'pication_r', 'pistack', 'saltbridge_n', 'saltbridge_p']

    if kwargs.get("skip"):
        id_batch = data_utils.trim_batch_ids(id_batch, mol_graph_ft)

    batch_total = len(id_batch)

    mol_sdf_ft = pdbbind_dir + "%s/%s_ligand.sdf"
    pl_data_file = pdbbind_dir + "/index/INDEX_general_PL_data.2020"
    print(pl_data_file)
    pl_data = {}

    with open(pl_data_file) as data_file_in:
        for line in data_file_in:
            if line[0] == "#":
                continue

            line = re.split(r"\s+", line)
            pl_data[line[0]] = float(line[3])

    for t_idx, pdb_id in enumerate(id_batch):
        mol_graph_file = mol_graph_ft % pdb_id

        print("%s: %s of %s" % (pdb_id, t_idx, batch_total))

        mol_sdf_file = mol_sdf_ft % (pdb_id, pdb_id)

        H_count = 0
        heavy_atom_pos = []
        mol_pos = []
        heavy_marker = []
        mol = None

        with Chem.SDMolSupplier(
            mol_sdf_file, removeHs=False, sanitize=False
        ) as sd_mol_in:
            for rdkit_mol in sd_mol_in:
                mol = rdkit_mol

        if mol is None:
            continue

        conformer = mol.GetConformer()

        for atom in mol.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            atom_x, atom_y, atom_z = (pos.x, pos.y, pos.z)
            mol_pos.append([atom_x, atom_y, atom_z])

            if atom.GetSymbol() == "H":
                H_count += 1
                heavy_marker.append(0)
                continue

            heavy_atom_pos.append([atom_x, atom_y, atom_z])
            heavy_marker.append(1)

        mol_y = np.zeros((len(mol_pos), len(INTERACTION_LABELS)))

        # if no ip file, complex is with peptide/nucleic acid, so skip it.
        if os.path.exists(ip_ft % pdb_id) == False:
            continue

        ip = pickle.load(open(ip_ft % pdb_id, "rb"))

        for record in ip:
            itype, interaction_xyz = record

            # list of atomic distances from interaction location to atoms in ligand
            # list items correspond in `pdb_data_distances` correspond to list items in `pdb_heavy_atom_pos`
            pdb_data_distances = np.array(
                [mol_gen_utils.get_distance(x, interaction_xyz) for x in heavy_atom_pos]
            )

            # list of atom indices corresponding to 'pdb_heavy_atom_pos', sorted by distance to interaction location
            sorted_pdb_data_indices = np.argsort(pdb_data_distances)

            # pication_r and pistack interactions located in the center of a ring
            # every member of that ring should be labeled with interaction
            if itype in ["pication_r", "pistack"]:
                min_distance = pdb_data_distances[sorted_pdb_data_indices[0]]

                # iterate through ligand atoms in order of distance to interaction location
                # if difference in shortest distance from current atom tdistance to interaction location is greater thatn 0.5...
                # ...we are out of the range of atoms that are members of the ring
                for a_i, atom_idx in enumerate(sorted_pdb_data_indices):
                    if pdb_data_distances[atom_idx] - min_distance > 0.5:
                        break

                # `a_i` nearest atoms to be labeled
                interacting_atoms = sorted_pdb_data_indices[:a_i]
            else:
                # not a ring-centered interaction, so only closest atom is to be labeled.
                interacting_atoms = [sorted_pdb_data_indices[0]]

            # update onehot vector representing interaction type corresponding to ligand atom
            for atom_idx in interacting_atoms:
                mol_y[atom_idx] = mol_gen_utils.one_hot_update(
                    INTERACTION_LABELS, mol_y[atom_idx], [itype]
                )

        try:
            # molecule = Chem.rdmolfiles.MolFromPDBBlock(pdb_file_content, removeHs=False)
            g = mol_gen_utils.generate_mol_graph(
                mol, torch.tensor(mol_y).float(), ATOM_LABELS
            )
            mol_pos = torch.tensor(mol_pos)
            heavy_marker = torch.tensor(heavy_marker)
            g.heavy = heavy_marker
            g.pos = mol_pos
            g.affinity = pl_data[pdb_id]
            pickle.dump(g, open(mol_graph_file, "wb"))
        except Exception as e:
            print(e, pdb_id)
