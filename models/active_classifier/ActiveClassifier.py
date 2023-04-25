import torch
import torch_geometric
import torch.nn as nn


class ActiveClassifier(torch.nn.Module):
    def __init__(self, poxel_model, poxel_params, mol_model, mol_params, in_dim):
        super(ActiveClassifier, self).__init__()
        self.pox_pooler = poxel_model(**poxel_params)
        self.mol_pooler = mol_model(**mol_params)

        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.linear3 = nn.Linear(in_dim, 512)
        self.linear4 = nn.Linear(512, 1)

        self.relu = nn.ReLU()

    def forward(self, pocket_batch, active_batch, decoy_batch):
        poxel_embeds = self.pox_pooler(pocket_batch)

        active_preds, active_embeds = self.mol_pooler(active_batch)
        print(active_embeds.shape)
        decoy_preds, decoy_embeds = self.mol_pooler(decoy_batch)
        # mol_atom_preds = torch.vstack((active_preds, decoy_preds))

        poxel_actives = torch.hstack((poxel_embeds, active_embeds))
        poxel_decoys = torch.hstack(
            (
                torch.cat([poxel_embeds] * len(decoy_embeds), dim=0),
                decoy_embeds.repeat_interleave(poxel_embeds.size(0), dim=0),
            )
        )

        all_embeds = torch.vstack((poxel_actives, poxel_decoys))

        x = self.linear1(all_embeds)
        x = self.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.linear2(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.4)

        x = self.linear3(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.3)

        x = self.linear4(x)
        # return x, mol_atom_preds
        return x
