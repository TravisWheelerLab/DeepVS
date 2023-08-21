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


# from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords
# from code.utils.get_distance import get_distance

import torch
from torch_geometric.data import Data


def get_bounding_box(point_coordinates, resolution):
    min_xyz, _ = torch.min(point_coordinates, 0)
    max_xyz, _ = torch.max(point_coordinates, 0)

    x_range = torch.arange(min_xyz[0], max_xyz[0]+resolution, resolution)
    y_range = torch.arange(min_xyz[1], max_xyz[1]+resolution, resolution)
    z_range = torch.arange(min_xyz[2], max_xyz[2]+resolution, resolution)

    return torch.cartesian_prod(x_range,y_range,z_range)

def assemble_pocket_graph(pdb_graph, 
                          voxel_coords, 
                          atom_neighbor_radius, 
                          vox_neighbor_count, 
                          labels,
                          interaction_profile: None,
                          interaction_labels: None):
    voxel_onehot = torch.zeros(len(labels))
    voxel_onehot[labels.index('VOXEL')] = 1

    nearest_atom_indices = torch.unique(torch.sort(knn(pdb_graph.pos, voxel_coords, vox_neighbor_count)[1])[0])

    atom_nodes = pdb_graph.x[nearest_atom_indices]
    atom_pos  = pdb_graph.pos[nearest_atom_indices]
    atom_edges = radius_graph(atom_pos, r=atom_neighbor_radius)

    graph_x = torch.vstack((atom_nodes, voxel_onehot.repeat(len(voxel_coords), 1)))
    graph_pos = torch.vstack((atom_pos, voxel_coords))

    vox_edges = knn(atom_pos, voxel_coords, vox_neighbor_count)
    vox_edges[0] += len(atom_nodes)

    edge_index = add_self_loops(to_undirected(torch.hstack((atom_edges, vox_edges))))[0]
    edge_weights = torch.norm(graph_pos[edge_index[0]] - graph_pos[edge_index[1]], dim=1).unsqueeze(dim=1)

    if interaction_profile:
        y = torch.zeros(len(graph_x), len(interaction_labels))
        ip_onehots = torch.zeros(len(interaction_profile), len(interaction_labels))

        for ip_index, record in enumerate(interaction_profile):
            ip_onehots[ip_index][interaction_labels.index(record[0])]=1

        # For every interaction in interaction profile, properly label closest voxel point
        ip_coords = torch.tensor([x[1] for x in interaction_profile]).float()

        interaction_edges = knn(voxel_coords, ip_coords, 2)
        interaction_edges[1] += len(atom_nodes)

        for onehot_idx, voxel_idx in zip(interaction_edges[0], interaction_edges[1]):
            y[voxel_idx] += ip_onehots[onehot_idx]
            y[voxel_idx][y[voxel_idx] > 1] = 1 

    full_pocket_graph = Data(x=graph_x, pos=graph_pos, edge_index=edge_index, edge_weights=edge_weights, y=y) 
    return full_pocket_graph


def generate_full_pockets(
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
        mol_graph_dir, data_dir, kwargs["mol_graph_file_template"]
    )
    ip_dir, ip_ft = data_utils.get_output_paths(
        ip_dir, data_dir, kwargs["interaction_profile_file_template"]
    )
    pocket_graph_dir, pocket_graph_ft = data_utils.get_output_paths(
        pocket_graph_dir, data_dir, kwargs["full_pocket_graph_file_template"]
    )

    POCKET_ATOM_LABELS = kwargs['POCKET_ATOM_LABELS']
    INTERACTION_LABELS = kwargs['INTERACTION_LABELS']

    print(mol_graph_dir, ip_dir, pocket_graph_dir)

    if os.path.exists(pocket_graph_dir) == False:
        os.makedirs(pocket_graph_dir)

    pocket_pdb_ft = pdbbind_dir + "%s/%s_protein.pdb"

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

        pdb_graph = pdb_utils.pdb_to_graph(pocket_pdb, POCKET_ATOM_LABELS) 
        mol_graph = pickle.load(open(mol_graph, "rb"))
        vox_coords = get_bounding_box(mol_graph.pos, resolution)

        full_pocket_graph = assemble_pocket_graph(pdb_graph, 
                                                  vox_coords, 
                                                  3.5, 
                                                  neighbor_count, 
                                                  POCKET_ATOM_LABELS, 
                                                  interaction_profile, 
                                                  INTERACTION_LABELS)
        
        pickle.dump(full_pocket_graph, open(pocket_graph_file, "wb"))

