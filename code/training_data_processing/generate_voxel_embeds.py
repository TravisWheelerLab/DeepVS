from importlib import import_module
import sys
from copy import deepcopy
import numpy as np
from torch_geometric.utils import subgraph 
from torch_geometric.data import Data
from code.utils.EdgeData import EdgeData
from code.utils.get_voxel_graphs import get_voxel_graphs
import torch
import pickle
import os
import code.utils.data_processing_utils as data_utils
import torch
import torch_geometric
import os

def generate_voxel_embeds(config: dict, 
                          id_batch: list, 
                          vox_encoder_model: str,
                          vox_encoder_weights: str,
                          pocket_graph_dir: str=None,
                          vox_embed_graph_dir: str=None,
                          data_dir: str=None,
                          **kwargs) -> None:
	
    pocket_graph_dir, pocket_graph_ft = data_utils.get_output_paths(pocket_graph_dir, data_dir, config['pocket_graph_file_template'])
    vox_embed_graph_dir, vox_embed_graph_ft = data_utils.get_output_paths(vox_embed_graph_dir, data_dir, config['vox_embed_graph_file_template'])

    if os.path.exists(vox_embed_graph_dir) == False:
    	os.makedirs(vox_embed_graph_dir)

    if kwargs.get('skip'):
        id_batch = data_utils.trim_batch_ids(id_batch, vox_embed_graph_ft)

    VoxEncoder = getattr(import_module(vox_encoder_model.replace("/", ".") % ), 'VoxEncoder')


    # for pdb_i, pdb_id in enumerate(id_batch):
    #     print("generating %s voxel embeds: %s of %s" % (pdb_id, pdb_i+1, batch_total))

    #     pocket_graph_file = pocket_graph_ft % pdb_id
    #     vox_embed_graph_file = vox_embed_graph_ft % pdb_id

    #     if os.path.exists(pocket_graph_file) == False:
    #         skip = True 

    #     if skip:
    #         continue

    #     if os.path.exists(interaction_file):
    #         interaction_profile = pickle.load(open(interaction_file, 'rb'))
    #     else:
    #         interaction_profile = []

    #     mol_graph = pickle.load(open(mol_graph, 'rb'))

    #     vox_coords = pocket_gen_utils.get_box_from_ligand_diagonal(ligand_atom_coords, 1.0, 1.0)
    #     pocket_graph = pocket_gen_utils.get_pocket_graph(config, pocket_pdb, vox_coords, interaction_profile, mol_graph)

    #     pickle.dump(pocket_graph, open(pocket_graph_file, 'wb'))





      







