import torch
import torch_geometric
from torch_geometric.nn import GCN2Conv, GraphNorm, GATv2Conv, InstanceNorm, LayerNorm
from torch_geometric.nn.models import MLP 
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import ReLU, SiLU, Sequential, Linear, BatchNorm1d, Dropout
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.nn import knn_graph

import torch.nn.functional as F
import sys

class SimpleGCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()

        self.edge_MLP = MLP([1, 32, 1])

        self.linear1 = torch.nn.Linear(feature_dim, hidden_dim)

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=6, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=6, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=6, concat=False)
        self.conv4 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=6, concat=False)

        self.norm1 = InstanceNorm(hidden_dim)
        self.norm2 = InstanceNorm(hidden_dim)
        self.norm3 = InstanceNorm(hidden_dim)
        self.norm4 = InstanceNorm(hidden_dim)

        self.embed_layer = Sequential(Linear(hidden_dim, hidden_dim),
                                      ReLU(),
                                      BatchNorm1d(hidden_dim),
                                      Linear(hidden_dim, hidden_dim))

        self.prediction_layer = Sequential(BatchNorm1d(hidden_dim),
                                           Linear(hidden_dim, out_dim))

        self.act = ReLU()
        self.relu = ReLU()
        self.dr = Dropout(0.25)

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch

        edge_index = add_self_loops(to_undirected(knn_graph(pos, k=10, batch=batch, loop=True)))[0]
        edge_weights = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(dim=1)
        edge_weights = self.relu(self.edge_MLP(edge_weights)).squeeze()

        x = self.linear1(x)

        h = self.act(x + self.conv1(x, edge_index, edge_weights))
        h = self.norm1(h)

        h = self.act(h + self.conv2(h, edge_index, edge_weights))
        h = self.norm2(h)  

        h = self.act(h + self.conv3(h, edge_index, edge_weights))
        h = self.norm3(h) 

        h = self.act(h + self.conv4(h, edge_index, edge_weights))
        h = self.norm4(h)

        h = self.embed_layer(h)
        interaction_pred = self.prediction_layer(h)

        return h, interaction_pred