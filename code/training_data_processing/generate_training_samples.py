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

def get_sample_graphs(pdb_graph, 
                      voxel_coords, 
                      neighbor_count, 
                      labels,
                      interaction_profile: None,
                      interaction_labels: None,
                      pdb_id: None):
    graph_list = []

    voxel_onehot = torch.zeros(len(labels))
    voxel_onehot[labels.index('VOXEL')] = 1

    ip_x = torch.zeros(len(interaction_profile), len(interaction_labels))
    ip_pos = torch.zeros((len(interaction_profile), 3))

    for ip_index, record in enumerate(interaction_profile):
        ip_x[ip_index][interaction_labels.index(record[0])]=1
        ip_pos[ip_index]=torch.tensor(record[1])

    interaction_voxel_edges = knn(voxel_coords, ip_pos, 2)
    voxel_atom_edges = knn(pdb_graph.pos, voxel_coords, neighbor_count) 

    for voxel_idx in torch.unique(torch.sort(interaction_voxel_edges[1])[0]):
        nearest_ip_mask = interaction_voxel_edges[1]==voxel_idx
        corresponding_ip_nodes = interaction_voxel_edges[0][nearest_ip_mask]

        graph_y = torch.sum(ip_x[corresponding_ip_nodes], dim=0)
        graph_y[graph_y > 1] = 1

        nearest_atoms_mask = voxel_atom_edges[0]==voxel_idx
        nearest_atoms = voxel_atom_edges[1][nearest_atoms_mask]

        graph_x = torch.vstack((voxel_onehot, pdb_graph.x[nearest_atoms]))
        graph_pos = torch.vstack((voxel_coords[voxel_idx], pdb_graph.pos[nearest_atoms]))
        sample_graph = Data(x=graph_x, pos=graph_pos, y=graph_y, pdb_id=pdb_id)

        graph_list.append(sample_graph)
    
    return graph_list


def generate_training_samples(
    id_batch: list,
    pdbbind_dir: str,
    mol_graph_dir: str = None,
    data_dir: str = None,
    ip_dir: str = None,
    training_sample_dir: str = None,
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
    training_sample_dir, training_sample_ft = data_utils.get_output_paths(
        training_sample_dir, data_dir, kwargs["training_sample_file_template"]
    )

    POCKET_ATOM_LABELS = kwargs['POCKET_ATOM_LABELS']
    INTERACTION_LABELS = kwargs['INTERACTION_LABELS']


    if os.path.exists(training_sample_dir) == False:
        os.makedirs(training_sample_dir)

    pocket_pdb_ft = pdbbind_dir + "%s/%s_protein.pdb"

    if kwargs.get("skip"):
        id_batch = data_utils.trim_batch_ids(id_batch, training_sample_ft)

    batch_total = len(id_batch)

    for pdb_i, pdb_id in enumerate(id_batch):
        print("!generating %s: %s of %s" % (pdb_id, pdb_i + 1, batch_total))

        mol_graph = mol_graph_ft % pdb_id
        pocket_pdb = pocket_pdb_ft % (pdb_id, pdb_id)
        interaction_file = ip_ft % pdb_id
        training_sample_file = training_sample_ft % pdb_id

        skip = False

        for f in [mol_graph, pocket_pdb]:
            if os.path.exists(f) == False:
                skip = True

        if skip:
            continue

        if os.path.exists(interaction_file):
            interaction_profile = pickle.load(open(interaction_file, "rb"))
        else:
            continue

        pdb_graph = pdb_utils.pdb_to_graph(pocket_pdb, POCKET_ATOM_LABELS) 
        mol_graph = pickle.load(open(mol_graph, "rb"))
        vox_coords = get_bounding_box(mol_graph.pos, resolution)

        sample_graphs = get_sample_graphs(pdb_graph, 
                                          vox_coords, 
                                          neighbor_count, 
                                          POCKET_ATOM_LABELS, 
                                          interaction_profile, 
                                          INTERACTION_LABELS,
                                          pdb_id)

        pickle.dump(sample_graphs, open(training_sample_file, "wb"))

