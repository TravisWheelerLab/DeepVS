# python deepvs.py generate_pocket_embeds params.yaml --in_file /xdisk/twheeler/jgaiser/deepvs3/experiments/litpcba_eval/data/ALDH1/ALDH1.pdb --pocket_bounds_file /xdisk/twheeler/jgaiser/deepvs3/experiments/litpcba_eval/data/ALDH1/ALDH1_bounds.txt
from copy import deepcopy
import os
import pickle
import sys
import numpy as np
import code.utils.data_processing_utils as data_utils
import code.utils.pdb_utils as pdb_utils
import code.utils.application_utils as app_utils 
import re 
import itertools

# from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords
# from code.utils.get_distance import get_distance

import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch


def generate_pocket_embeds(
    in_file: str=None,
    out_file: str=None,
    pocket_bounds_file: str=None,
    vox_embedder_model: str=None,
    vox_embedder_weights: str=None,
    resolution: float = 1.0,
    neighbor_count: int = 10,
    **kwargs
) -> None:

# ------------ PDBBIND TESTING ---------------- #
    validation_ids_flat = []
    validation_ids = pickle.load(open('/xdisk/twheeler/jgaiser/deepvs3/training_data/validation_ids.pkl', 'rb'))
    embed_out_ft = '/xdisk/twheeler/jgaiser/deepvs3/experiments/pdbbind_validation/app_embeds/%s.pt'

    for row in validation_ids:
        for pdb_id in row:
            validation_ids_flat.append(pdb_id)

    pdb_ft = kwargs['pdbbind_dir'] + '%s/%s_protein.pdb'
    mol_ft = kwargs['data_dir'] + 'graph_data/mol_graphs/%s_mol.pkl'
# ------------ /PDBBIND TESTING ---------------- #

    POCKET_ATOM_LABELS = kwargs['POCKET_ATOM_LABELS']
    # pdb_graph = pdb_utils.pdb_to_graph(in_file, POCKET_ATOM_LABELS) 

    vox_embedder_model = data_utils.interpolate_root(vox_embedder_model, kwargs['root_dir'])
    vox_embedder_weights = data_utils.interpolate_root(vox_embedder_weights, kwargs['root_dir'])

    poxel_agg_model = data_utils.interpolate_root(kwargs['poxel_aggregator_model'], kwargs['root_dir'])
    poxel_agg_weights = data_utils.interpolate_root(kwargs['poxel_aggregator_weights'], kwargs['root_dir'])

    PoxAggregator = data_utils.load_class_from_file(poxel_agg_model)
    VoxEmbedder = data_utils.load_class_from_file(vox_embedder_model)

    vox_embedder = VoxEmbedder(**kwargs['vox_embedder_hyperparams'])
    vox_embedder.load_state_dict(torch.load(vox_embedder_weights, map_location=torch.device('cpu')))

    pox_aggregator = PoxAggregator(**kwargs['poxel_aggregator_hyperparams'])
    pox_aggregator.load_state_dict(torch.load(poxel_agg_weights, map_location=torch.device('cpu')))

    pox_aggregator.eval()
    vox_embedder.eval()

    voxel_onehot = [0 for _ in POCKET_ATOM_LABELS]
    voxel_onehot[POCKET_ATOM_LABELS.index('VOXEL')] = 1
    voxel_onehot = torch.tensor(voxel_onehot)

    # poxel_list = []

    # with open(pocket_bounds_file, 'r') as pocket_bounds_in:
        # embed_number = 0

        # for line in pocket_bounds_in:
            # line = line.rstrip()
# ------------ PDBBIND TESTING ---------------- #
    for pdb_id in validation_ids_flat:
            print(pdb_id)
            if os.path.exists(embed_out_ft % pdb_id):
                 continue
            # embed_number += 1
            # embed_filename = pocket_bounds_file.split('.')[0] + "_%s.pkl" % embed_number 
            
            # corner_coords = torch.tensor([float(x) for x in re.split(r'\s+', line)])

            # x_range = torch.arange(corner_coords[0], corner_coords[3]+resolution, resolution)
            # y_range = torch.arange(corner_coords[1], corner_coords[4]+resolution, resolution)
            # z_range = torch.arange(corner_coords[2], corner_coords[5]+resolution, resolution)
            
            # voxel_coords = torch.cartesian_prod(x_range,y_range,z_range)
# ------------ PDBBIND TESTING ---------------- #
            pdb_graph = pdb_utils.pdb_to_graph(pdb_ft % (pdb_id, pdb_id), POCKET_ATOM_LABELS) 
            if os.path.exists(mol_ft % pdb_id) == False:
                 continue
            mol_graph = pickle.load(open(mol_ft % pdb_id, "rb"))
            voxel_coords = app_utils.get_bounding_box(mol_graph.pos, resolution)
# ------------ /PDBBIND TESTING ---------------- #

            poxel_graph = app_utils.get_vox_embed(pdb_graph, 
                                                  vox_embedder,
                                                  voxel_coords, 
                                                  neighbor_count, 
                                                  POCKET_ATOM_LABELS)

            with torch.no_grad():  
                poxel_embed = pox_aggregator(poxel_graph).squeeze()
                torch.save(poxel_embed, embed_out_ft % pdb_id)
            # poxel_list.append(poxel_embed)
# ------------ PDBBIND TESTING ---------------- #
            # torch.save(poxel_embed, out_dir + "%s.pt" % pdb_id) 


    # torch.save(torch.vstack(poxel_list), out_file)
   