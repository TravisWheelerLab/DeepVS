import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class ActiveClassifier(torch.nn.Module):
    def __init__(self, poxel_model, mol_embed_model, mol_agg_model, mlp_model, in_dim, hidden_dim, res_layers=3):
        super(ActiveClassifier, self).__init__()
        self.pox_agg = poxel_model[0](**poxel_model[1])
        self.mol_embedder = mol_embed_model[0](**mol_embed_model[1])
        self.mol_agg = mol_agg_model[0](**mol_agg_model[1])
        self.classifier_mlp = mlp_model[0](**mlp_model[1])

        # if 'weights' in mol_embed_model[1]:
        #     self.mol_embedder.load_state_dict(torch.load(mol_embed_model[1]['weights']))

        # self.res_layers = []

        # for _ in range(res_layers):
        #     self.res_layers.append(ResLayer(hidden_dim, dr=0.25))

        # self.linear_in = nn.Linear(in_dim, hidden_dim)
        # self.act = nn.ReLU()

        # self.res_layers = nn.Sequential(*self.res_layers)
        self.linear_out = nn.Linear(hidden_dim, 1)
    
    def get_pox_model(self):
        return self.pox_agg

    def get_mol_embedder(self):
        return self.mol_embedder

    def get_mol_agg(self):
        return self.mol_agg

    def get_classifier_mlp(self):
        return self.classifier_mlp


    def forward(self, pocket_batch, active_batch, decoy_pockets=None, decoy_batch=None):
        batch_size = len(pocket_batch)
        poxel_embeds = self.pox_agg(pocket_batch)

        if decoy_pockets:
            decoy_poxel_embeds = self.pox_agg(decoy_pockets)

        active_atom_embeds, mol_preds = self.mol_embedder(active_batch)
        active_batch = deepcopy(active_batch)
        active_batch.x = active_atom_embeds
        active_mol_embeds = self.mol_agg(active_batch)

        if decoy_batch:
            decoy_atom_embeds, decoy_mol_preds = self.mol_embedder(decoy_batch)
            decoy_batch = deepcopy(decoy_batch)
            decoy_batch.x = decoy_atom_embeds
            decoy_mol_embeds = self.mol_agg(decoy_batch)
            all_mol_embeds = torch.vstack((active_mol_embeds,
                                           decoy_mol_embeds))

            mol_preds = torch.vstack((mol_preds, 
                                      decoy_mol_preds))  

        else:
            x = torch.arange(batch_size)
            rotations = []

            for i in range(len(x)):
                rotations.append(torch.roll(x, shifts=i))

            mol_embed_indices = torch.cat(rotations)
            all_mol_embeds =  active_mol_embeds[mol_embed_indices]                                       

        y = torch.zeros(len(all_mol_embeds))
        y[:batch_size] = 1
     
        if decoy_pockets:
            poxel_embeds = torch.vstack((poxel_embeds, decoy_poxel_embeds))
        else:
            poxel_embeds = poxel_embeds.repeat(int(len(all_mol_embeds)/batch_size)+1, 1)
            poxel_embeds = poxel_embeds[:-(len(poxel_embeds) - len(all_mol_embeds))]

        if len(all_mol_embeds) != len(poxel_embeds):
            return None, None, None, None, None

        all_embeds = torch.hstack((poxel_embeds, all_mol_embeds))

        # x = self.res_layers(all_embeds)
        # x = self.linear_out(x)
        x = self.classifier_mlp(all_embeds)
        
        return x,y, mol_preds, poxel_embeds, all_mol_embeds
