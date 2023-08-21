import sys
from copy import deepcopy
import numpy as np
from torch_geometric.utils import subgraph 
from torch_geometric.data import Data
from code.utils.EdgeData import EdgeData
import code.utils.data_processing_utils as data_utils
import torch
import pickle
import os

def generate_training_samples(id_batch: list, 
                              training_samples_dir: str=None,
                              data_dir: str=None,
                              pocket_graph_dir: str=None,
                              neighbor_count: int=10,
                              **kwargs) -> None:

    training_samples_dir, training_sample_ft = data_utils.get_output_paths(training_samples_dir, 
                                                                            data_dir, 
                                                                            kwargs['training_sample_file_template'])

    pocket_graph_dir, pocket_graph_ft = data_utils.get_output_paths(pocket_graph_dir,
                                                                    data_dir,
                                                                    kwargs['pocket_graph_file_template']) 

    if os.path.exists(training_samples_dir)==False:
        os.makedirs(training_samples_dir)

    POCKET_ATOM_LABELS = config['POCKET_ATOM_LABELS']
    INTERACTION_LABELS = config['INTERACTION_LABELS']
    voxel_label_index = POCKET_ATOM_LABELS.index('VOXEL')

    if kwargs.get('skip'):
        id_batch = data_utils.trim_batch_ids(id_batch, training_sample_ft)

    batch_total = len(id_batch)

    sample_total = 0

    for pdb_i, pdb_id in enumerate(id_batch): 
        print("%s: %s of %s" % (pdb_id, pdb_i+1, batch_total))
        pocket_file = pocket_graph_ft % pdb_id

        if os.path.exists(pocket_file) == False:
            continue

        pocket_graph = pickle.load(open(pocket_file, 'rb'))
        voxel_indices = torch.where(pocket_graph.x[:, voxel_label_index]==1)[0]

        # Copy of position data for finding neighboring protein atoms
        # voxel pos set to infinity so they are effectively discounted
        pos_masked = deepcopy(pocket_graph.pos)
        pos_masked[voxel_indices] = torch.tensor([float('inf')]*3)

        training_sample_file = training_sample_ft % pdb_id

        graph_list = []

        occupied_voxels = torch.where(pocket_graph.contact_map != -1)[0]
        interacting_voxels = torch.where(torch.sum(pocket_graph.y, dim=1) > 0)[0]
        filtered_voxels = torch.sort(torch.cat((occupied_voxels, interacting_voxels)).unique())[0]
        filtered_atoms = torch.tensor([])

        for voxel_index in filtered_voxels:
            edge_data = EdgeData(EDGE_LABELS)

            neighbor_atom_indices = torch.sort(torch.cdist(pocket_graph.pos[voxel_index].unsqueeze(0), 
                                                           pos_masked)[0])[1][:neighbor_count]

            sample_graph_indices = torch.hstack((voxel_index, neighbor_atom_indices))

            for n_i in range(sample_graph_indices.size(0)):
                node_distances = torch.cdist(pocket_graph.pos[sample_graph_indices[n_i]].unsqueeze(0),
                                             pocket_graph.pos[sample_graph_indices[n_i:]])[0]

                for n_j, d in enumerate(node_distances):
                    if n_j==0:
                        if n_i==0:
                            edge_label = 'voxel-self'
                        else:
                            edge_label = 'atom-self'
                    else:
                        if n_i==0:
                            edge_label = 'atom-voxel'
                        else:
                            edge_label = 'atom-atom'

                    edge_data.add_edge(n_i, n_i+n_j, d.item(), edge_label)

            g_edge_index, g_edge_attr, g_edge_labels = edge_data.get_data()

            sample_graph = Data(x=pocket_graph.x[sample_graph_indices],
                                y=pocket_graph.y[sample_graph_indices[0]].unsqueeze(0),
                                pos=pocket_graph.pos[sample_graph_indices], 
                                contact_map=pocket_graph.contact_map[sample_graph_indices[0]],
                                beta=pocket_graph.beta[sample_graph_indices],
                                pdb_id=pdb_id,
                                edge_index=torch.tensor(g_edge_index, dtype=torch.long),
                                edge_attr=torch.tensor(g_edge_attr, dtype=torch.float),
                                edge_labels=torch.tensor(g_edge_labels, dtype=torch.long))

            graph_list.append(sample_graph) 

        pickle.dump(graph_list, open(training_sample_file, 'wb'))
