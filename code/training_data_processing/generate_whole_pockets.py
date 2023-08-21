from copy import deepcopy
import os
import pickle
import sys
import numpy as np
import code.utils.data_processing_utils as data_utils
import code.utils.pdb_utils as pdb_utils
import code.utils.pocket_gen_utils as pocket_gen_utils

# from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords
# from code.utils.get_distance import get_distance

import torch
from torch_geometric.data import Data


def generate_full_pockets(
    config: dict,
    id_batch: list,
    pdbbind_dir: str,
    mol_graph_dir: str = None,
    data_dir: str = None,
    ip_dir: str = None,
    pocket_graph_dir: str = None,
    resolution: float = 1.0,
    neighbor_count: int = 10,
    **kwargs
) -> None:
    mol_graph_dir, mol_graph_ft = data_utils.get_output_paths(
        mol_graph_dir, data_dir, config["mol_graph_file_template"]
    )
    ip_dir, ip_ft = data_utils.get_output_paths(
        ip_dir, data_dir, config["interaction_profile_file_template"]
    )
    pocket_graph_dir, pocket_graph_ft = data_utils.get_output_paths(
        pocket_graph_dir, data_dir, config["pocket_graph_file_template"]
    )

    print(mol_graph_dir, ip_dir, pocket_graph_dir)

    if os.path.exists(pocket_graph_dir) == False:
        os.makedirs(pocket_graph_dir)

    pocket_pdb_ft = pdbbind_dir + "%s/%s_pocket.pdb"

    if kwargs.get("skip"):
        id_batch = data_utils.trim_batch_ids(id_batch, pocket_graph_ft)

    batch_total = len(id_batch)

    for pdb_i, pdb_id in enumerate(id_batch):
        print("generating %s: %s of %s" % (pdb_id, pdb_i + 1, batch_total))

        mol_graph = mol_graph_ft % pdb_id
        pocket_pdb = pocket_pdb_ft % (pdb_id, pdb_id)
        interaction_file = ip_ft % pdb_id
        pocket_graph_file = pocket_graph_ft % pdb_id

        skip = False

        for f in [mol_graph, pocket_pdb]:
            if os.path.exists(f) == False:
                skip = True

        if skip:
            continue

        if os.path.exists(interaction_file):
            interaction_profile = pickle.load(open(interaction_file, "rb"))
        else:
            interaction_profile = []

        mol_graph = pickle.load(open(mol_graph, "rb"))

        vox_coords = pocket_gen_utils.get_box_from_ligand_diagonal(
            mol_graph.pos.tolist(), 1.0, resolution
        )

        pocket_graph = pocket_gen_utils.get_pocket_graph(
            config, pocket_pdb, vox_coords, interaction_profile, mol_graph
        )

        pocket_graph.resolution = resolution

        pickle.dump(pocket_graph, open(pocket_graph_file, "wb"))
