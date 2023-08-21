import torch
import torch_geometric
from torch_geometric.nn import GCN2Conv, LayerNorm, global_mean_pool, InstanceNorm
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn.models import MLP 

import torch.nn.functional as F

class MolGCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()

        mlp_h = MLP([hidden_dim, 1])
        mlp_theta = MLP([hidden_dim, hidden_dim])

        self.linear1 = torch.nn.Linear(feature_dim, hidden_dim)

        self.conv1 = GCN2Conv(hidden_dim, 0.2)
        self.conv2 = GCN2Conv(hidden_dim, 0.2)
        self.conv3 = GCN2Conv(hidden_dim, 0.2)
        self.conv4 = GCN2Conv(hidden_dim, 0.2)

        self.dropout = torch.nn.Dropout(p=0.25)
        self.batchnorm = torch_geometric.nn.BatchNorm(hidden_dim)
        self.norm1 = InstanceNorm(hidden_dim)
        self.norm2 = InstanceNorm(hidden_dim)
        self.norm3 = InstanceNorm(hidden_dim)
        self.norm4 = InstanceNorm(hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)
        self.aggr = AttentionalAggregation(mlp_h, mlp_theta)

        self.act = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_weights, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.linear1(x)

        h1 = self.conv1(x, x, edge_index)
        h1 = self.norm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1, x, edge_index)
        h2 = self.norm2(h2)  
        h2 = self.act(h2+h1)

        h3 = self.conv3(h2, x, edge_index)
        h3 = self.norm3(h3) 
        h3 = self.act(h3)

        h4 = self.conv4(h3, x, edge_index)
        h4 = self.norm4(h4)
        h4 = self.act(h4+h2)

        atom_embeds = F.normalize(h4, dim=-1, p=2)
        atom_preds = self.linear2(h4)
        mol_embed = self.aggr(h4, index=batch)

        return atom_embeds, atom_preds, mol_embed
        # return  #normalize(x, dim=-1, p=2), self.atom_classifier(x), self.lin2(out)