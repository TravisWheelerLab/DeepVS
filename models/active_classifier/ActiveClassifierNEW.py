import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class ActiveClassifier(torch.nn.Module):
    def __init__(self, poxel_model, mol_embed_model, mol_agg_model, in_dim, hidden_dim):
        super(ActiveClassifier, self).__init__()
        self.pox_agg = poxel_model[0](**poxel_model[1])
        self.mol_embedder = mol_embed_model[0](**mol_embed_model[1])
        self.mol_agg = mol_agg_model[0](**mol_agg_model[1])

        if 'model_weights' in mol_embed_model[1]:
            self.mol_embedder.load_state_dict(torch.load(mol_embed_model[1]['model_weights']))

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear4 = nn.Linear(int(hidden_dim/2), 1)

        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(int(hidden_dim/2))

        self.relu = nn.ReLU()

    def forward(self, pocket_batch, active_batch, decoy_batch=None):
        poxel_embeds = self.pox_agg(pocket_batch)

        active_atom_embeds, _, _ = self.mol_embedder(active_batch)
        active_batch = deepcopy(active_batch)
        active_batch.x = active_atom_embeds
        active_mol_embeds = self.mol_agg(active_batch)

        if decoy_batch:
            decoy_atom_embeds, _, _ = self.mol_embedder(decoy_batch)
            decoy_batch = deepcopy(decoy_batch)
            decoy_batch.x = decoy_atom_embeds
            decoy_mol_embeds = self.mol_agg(decoy_batch)

        poxel_actives = torch.hstack((poxel_embeds, active_mol_embeds))
        poxel_decoys = torch.hstack(
            (
                torch.cat([poxel_embeds] * len(decoy_mol_embeds), dim=0),
                decoy_mol_embeds.repeat_interleave(poxel_embeds.size(0), dim=0),
            )
        )

        all_embeds = torch.vstack((poxel_actives, poxel_decoys))
        y = torch.zeros(len(all_embeds))
        y[:len(poxel_actives)] = 1

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class ActiveClassifier(torch.nn.Module):
    def __init__(self, poxel_model, mol_embed_model, mol_agg_model, in_dim, hidden_dim):
        super(ActiveClassifier, self).__init__()
        self.pox_agg = poxel_model[0](**poxel_model[1])
        self.mol_embedder = mol_embed_model[0](**mol_embed_model[1])
        self.mol_agg = mol_agg_model[0](**mol_agg_model[1])

        if 'model_weights' in mol_embed_model[1]:
            self.mol_embedder.load_state_dict(torch.load(mol_embed_model[1]['model_weights']))

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, int(hidden_dim/2))

import torch
import torch_geometric
from torch_geometric.nn import GCN2Conv, GCNConv, GraphConv
from torch_geometric.nn import SAGPooling
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.pool import ASAPooling

from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation
from torch_geometric.nn.models import MLP 
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected


import torch.nn.functional as F


class PoxelGCN(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()

        self.conv1 = GCN2Conv(hidden_dim, 0.2)
        self.conv2 = GCN2Conv(hidden_dim, 0.2)
        self.conv3 = GCN2Conv(hidden_dim, 0.2)
        self.conv4 = GCN2Conv(hidden_dim, 0.2)

        self.pool1 = ASAPooling(hidden_dim, ratio=0.15, GNN=GraphConv, add_self_loops=False)
        self.pool2 = ASAPooling(hidden_dim, ratio=0.25, GNN=GraphConv, add_self_loops=False)
        self.pool3 = ASAPooling(hidden_dim, ratio=0.5, GNN=GraphConv, add_self_loops=False)

        self.edge_MLP0 = MLP([1, 32, 1])
        self.edge_MLP1 = MLP([1, 32, 1])
        self.edge_MLP2 = MLP([1, 32, 1])
        self.edge_MLP3 = MLP([1, 32, 1])

        self.mean_aggr = MeanAggregation()
        self.max_aggr = MaxAggregation()

        gate_nn = MLP([hidden_dim, 1], act="relu")
        nn = MLP([hidden_dim, out_dim], act="relu")
        self.global_pool = AttentionalAggregation(gate_nn, nn)

    def forward(self, data):
        x, pos, batch = (
            data.x,
            data.pos,
            data.batch,
        )

        edge_index = to_undirected(knn_graph(pos, k=6, batch=batch, loop=True))
        edge_weights = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(dim=1)
        edge_weights = F.relu(self.edge_MLP0(edge_weights).squeeze())

        h = self.conv1(x, x, edge_index, edge_weights)
        h = F.relu(h)
        h, _, _, batch, k_index = self.pool1(h, edge_index, edge_weight=edge_weights, batch=batch)
        x = x[k_index]
        pos = pos[k_index]

        edge_index = to_undirected(knn_graph(pos, k=6, batch=batch, loop=True))
        edge_weights = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(dim=1)
        edge_weights = F.relu(self.edge_MLP1(edge_weights).squeeze())

        readout1 = torch.hstack((self.mean_aggr(h, index=batch),
                                 self.max_aggr(h, index=batch)))


        h = self.conv2(h, x, edge_index, edge_weights)
        h = F.relu(h)
        h, _, _, batch, k_index = self.pool2(h, edge_index, edge_weight=edge_weights, batch=batch)
        x = x[k_index]
        pos = pos[k_index]

        edge_index = to_undirected(knn_graph(pos, k=6, batch=batch, loop=True))
        edge_weights = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(dim=1)
        edge_weights = F.relu(self.edge_MLP2(edge_weights).squeeze())

        readout2 = torch.hstack((self.mean_aggr(h, index=batch),
                                 self.max_aggr(h, index=batch)))

        h = self.conv3(h, x, edge_index, edge_weights)
        h = F.relu(h)
        h, _, _, batch, k_index = self.pool3(h, edge_index, edge_weight=edge_weights, batch=batch)
        x = x[k_index]
        pos = pos[k_index]

        edge_index = to_undirected(knn_graph(pos, k=6, batch=batch, loop=True))
        edge_weights = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(dim=1)
        edge_weights = F.relu(self.edge_MLP3(edge_weights).squeeze())

        h = self.conv4(h, x, edge_index, edge_weights)
        h = F.relu(h)

        h = torch.hstack([self.global_pool(h, index=batch),
                          readout2,
                          readout1])

        return h

import torch
import torch_geometric
from torch_geometric.nn import GCN2Conv, LayerNorm, global_mean_pool, InstanceNorm, GraphConv
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn.pool import ASAPooling

from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation
from torch_geometric.nn.models import MLP 

import torch.nn.functional as F

class MolAggregator(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()

        mlp_h = MLP([hidden_dim, 1])
        mlp_theta = MLP([hidden_dim, hidden_dim])

        self.conv1 = GCN2Conv(hidden_dim, 0.2)
        self.conv2 = GCN2Conv(hidden_dim, 0.2)

        self.norm1 = InstanceNorm(hidden_dim)

        self.pool1 = ASAPooling(feature_dim, ratio=0.5, GNN=GraphConv, add_self_loops=True)
        self.pool2 = ASAPooling(hidden_dim, ratio=0.5, GNN=GraphConv, add_self_loops=True)

        self.mean_aggr = MeanAggregation()
        self.max_aggr = MaxAggregation()

        self.dropout = torch.nn.Dropout(p=0.25)
        self.batchnorm = torch_geometric.nn.BatchNorm(hidden_dim)
        self.norm1 = InstanceNorm(hidden_dim)
        self.norm2 = InstanceNorm(hidden_dim)
        self.mol_aggr = AttentionalAggregation(mlp_h, mlp_theta)

        self.act = torch.nn.ReLU()

        self.readout_pool1 = AttentionalAggregation(MLP([hidden_dim, 1], act="relu"), 
                                                    MLP([hidden_dim, hidden_dim*2], act="relu"))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        readout1 = torch.hstack((self.mean_aggr(x, index=batch),
                                 self.max_aggr(x, index=batch)))

        h, edge_index, _, batch, k_index = self.pool1(x, edge_index, batch=batch)
        x = x[k_index]
        h = self.conv1(h, x, edge_index)
        h = self.norm1(h)
        h = self.act(h)

        readout2 = torch.hstack((self.mean_aggr(h, index=batch),
                                 self.max_aggr(h, index=batch)))


        h, edge_index, _, batch, k_index = self.pool2(h, edge_index, batch=batch)
        x = x[k_index]
        h = self.conv2(h, x, edge_index)
        h = self.norm2(h)
        h = self.act(h)

        h = torch.hstack([self.mol_aggr(h, index=batch),
                          readout2,
                          readout1])

        return h 

        self.linear4 = nn.Linear(int(hidden_dim/2), 1)

        self.bn1 = nn.BatchNorm1d(in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(int(hidden_dim/2))

        self.relu = nn.ReLU()

    def forward(self, pocket_batch, active_batch, decoy_batch=None):
        batch_size = len(pocket_batch)
        poxel_embeds = self.pox_agg(pocket_batch)

        active_atom_embeds, _, _ = self.mol_embedder(active_batch)
        active_batch = deepcopy(active_batch)
        active_batch.x = active_atom_embeds
        active_mol_embeds = self.mol_agg(active_batch)

        if decoy_batch:
            decoy_atom_embeds, _, _ = self.mol_embedder(decoy_batch)
            decoy_batch = deepcopy(decoy_batch)
            decoy_batch.x = decoy_atom_embeds
            decoy_mol_embeds = self.mol_agg(decoy_batch)
            all_mol_embeds = torch.vstack((active_mol_embeds,
                                           decoy_mol_embeds.repeat_interleave(poxel_embeds.size(0), dim=0)))
            y = torch.zeros(len(all_mol_embeds))
            y[:batch_size] = 1
        else:
            all_mol_embeds = active_mol_embeds.repeat_interleave(poxel_embeds.size(0), dim=0)
            y = torch.zeros(len(all_mol_embeds))

            for y_i in range(batch_size):
                y[y_i * batch_size + y_i] = 1
     
        poxel_embeds = poxel_embeds.repeat(int(len(all_mol_embeds)/len(pocket_batch)), 1)
        all_embeds = torch.hstack((poxel_embeds, all_mol_embeds))

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
        return x,y


        poxel_actives = torch.hstack((poxel_embeds, active_mol_embeds))
        poxel_decoys = torch.hstack(
            (
                torch.cat([poxel_embeds] * len(decoy_mol_embeds), dim=0),
                decoy_mol_embeds.repeat_interleave(poxel_embeds.size(0), dim=0),
            )
        )

        all_embeds = torch.vstack((poxel_actives, poxel_decoys))
        y = torch.zeros(len(all_embeds))
        y[:len(poxel_actives)] = 1

        all_embeds = self.bn1(all_embeds)

        x = self.linear1(all_embeds)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.25)

        x = self.linear2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.25)

        x = self.linear3(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.25)

        x = self.linear4(x)
        # return x, mol_atom_preds
        return x, y
