# python deepvs.py generate_pocket_embeds params.yaml --pdb_file /xdisk/twheeler/jgaiser/deepvs3/experiments/litpcba_eval/data/ALDH1/ALDH1.pdb --pocket_bounds_file /xdisk/twheeler/jgaiser/deepvs3/experiments/litpcba_eval/data/ALDH1/ALDH1_bounds.txt
from copy import deepcopy
import os
import pickle
import sys
import numpy as np
import code.utils.data_processing_utils as data_utils
import code.utils.pdb_utils as pdb_utils
import code.utils.application_utils as app_utils 
import code.utils.mol_gen_utils as mol_gen_utils
import re 
import itertools
from rdkit import Chem

# from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords
# from code.utils.get_distance import get_distance

import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch


def generate_mol_embeds(
    in_file: str=None,
    out_file: str=None,
    mol_embedder_model: str=None,
    mol_embedder_weights: str=None,
    mol_agg_model: str=None,
    mol_agg_weights: str=None,
    **kwargs
) -> None:
    # TESTING
    mol_embeds_out_ft = '/xdisk/twheeler/jgaiser/deepvs3/experiments/pdbbind_validation/mol_embeds/%s.pt'
    val_ids = '/xdisk/twheeler/jgaiser/deepvs3/training_data/validation_ids.pkl' 
    mol_ft = '/xdisk/twheeler/jgaiser/deepvs3/training_data/graph_data/mol_smi_graphs/%s_mol.pt' 
    #/TESTING

    mol_embedder_model = data_utils.interpolate_root(mol_embedder_model, kwargs['root_dir'])
    mol_embedder_weights = data_utils.interpolate_root(mol_embedder_weights, kwargs['root_dir'])

    mol_agg_model = data_utils.interpolate_root(kwargs['mol_aggregator_model'], kwargs['root_dir'])
    mol_agg_weights = data_utils.interpolate_root(kwargs['mol_aggregator_weights'], kwargs['root_dir'])

    MolAggregator = data_utils.load_class_from_file(mol_agg_model)
    MolEmbedder = data_utils.load_class_from_file(mol_embedder_model)

    mol_embedder = MolEmbedder(**kwargs['mol_embedder_hyperparams'])
    mol_embedder.load_state_dict(torch.load(mol_embedder_weights, map_location='cpu'))

    mol_aggregator = MolAggregator(**kwargs['mol_aggregator_hyperparams'])
    mol_aggregator.load_state_dict(torch.load(mol_agg_weights, map_location='cpu'))

    mol_embedder.eval()
    mol_aggregator.eval()

    mol_embed_list = []

    # with open(in_file, 'rbU') as smi_in:
    #     line_count = sum(1 for _ in smi_in)

    # with open(in_file, 'r') as smi_in:
    #TESTING
    for row in pickle.load(open(val_ids, 'rb')):
        for pdb_id in row:
    #/TESTING
    #         line_number = 1
    #         for line in smi_in:
    #             line = line.rstrip()
    #             smile_string = re.split(r'\s+', line)[0]
            #TESTING
            if os.path.exists(mol_ft % pdb_id) == False:
                continue
            smile_string = (torch.load(mol_ft % pdb_id).smiles)
            #/TESTING
            with torch.no_grad():
                g = app_utils.graph_from_smile(smile_string)
                atom_embeds, _ = mol_embedder(g)
                g.x = atom_embeds 
                mol_embed = mol_aggregator(g).squeeze()
                #TESTING
                torch.save(mol_embed, mol_embeds_out_ft % pdb_id)
                print(pdb_id)
                #/TESTING
    #             mol_embed_list.append(mol_embed)
    #             print("%s of %s" % (line_number, line_count))
    #             line_number += 1

    # # torch.save(torch.vstack(mol_embed_list), out_file)



