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

#TEST
from glob import glob

# from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords
# from code.utils.get_distance import get_distance

import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch

sigmoid = torch.nn.Sigmoid()

def score_embeds(
    pocket_file_in: str=None,
    mol_file_in: str=None,
    out_file: str=None,
    classifier_mlp_model: str=None,
    classifier_mlp_hyperparams: dict=None,
    classifier_mlp_weights:str=None,
    **kwargs
) -> None:
    #TEST
    validation_ids = pickle.load(open('/xdisk/twheeler/jgaiser/deepvs3/training_data/validation_ids.pkl', 'rb'))
    pox_embed_ft = '/xdisk/twheeler/jgaiser/deepvs3/experiments/pdbbind_validation/app_embeds/%s.pt'
    mol_embed_ft = '/xdisk/twheeler/jgaiser/deepvs3/experiments/pdbbind_validation/mol_embeds/%s.pt'
    #/TEST

    classifier_mlp_model = data_utils.interpolate_root(classifier_mlp_model, kwargs['root_dir'])
    classifier_mlp_weights = data_utils.interpolate_root(classifier_mlp_weights, kwargs['root_dir'])

    ClassifierMLP = data_utils.load_class_from_file(classifier_mlp_model)
    classifier_mlp = ClassifierMLP(**classifier_mlp_hyperparams)
    classifier_mlp.load_state_dict(torch.load(classifier_mlp_weights, map_location='cpu'))
    classifier_mlp.eval()
    
    # pocket_embeds = torch.load(pocket_file_in)
    # mol_embeds = torch.load(mol_file_in)
    # mol_embeds = mol_embeds[:20000]

    # # mlp_input = torch.hstack((pocket_embeds[1].repeat(len(mol_embeds), 1), mol_embeds))
    # mlp_input = torch.hstack((torch.randn(1024).repeat(len(mol_embeds), 1), mol_embeds))
    # classifier_out = sigmoid(classifier_mlp(mlp_input))
    # print(classifier_out[:100])

    #TEST
    pox_embeds = []
    mol_embeds = []

    for row in validation_ids:
        row_trimmed = [x for x in row if os.path.exists(pox_embed_ft % x)]

        pox_embeds.append(torch.vstack([torch.load(pox_embed_ft % x) for x in row_trimmed]))
        mol_embeds.append(torch.vstack([torch.load(mol_embed_ft % x) for x in row_trimmed]))

    for i in range(len(pox_embeds)):
        negative_mols = None

        for j in range(len(pox_embeds)):
            if i==j:
                positive_mols = mol_embeds[j]
            else:
                negative_mols = torch.vstack((negative_mols, mol_embeds[j])) if negative_mols is not None else mol_embeds[j]


        negative_poxels = pox_embeds[i].repeat(int(len(negative_mols)/len(pox_embeds[i]))+1, 1)[:len(negative_mols)]

        positive_samples = torch.hstack((pox_embeds[i], positive_mols))
        negative_samples = torch.hstack((negative_poxels, negative_mols))

        with torch.no_grad():
            pos_out = sigmoid(classifier_mlp(positive_samples))
            neg_out = sigmoid(classifier_mlp(negative_samples))

        print(torch.mean(pos_out))
        print(torch.mean(neg_out))
        print('-------')
     
    #/TEST