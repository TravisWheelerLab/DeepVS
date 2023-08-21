import torch
import sys
import torch_geometric
from torch_geometric.nn import GCN2Conv, GCNConv, GraphConv, GATv2Conv
from torch_geometric.nn import SAGPooling
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.pool import ASAPooling, SAGPooling, TopKPooling

from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation
from torch_geometric.nn.models import MLP 
from torch_geometric.nn import knn_graph, Linear as Lin,  GraphNorm, InstanceNorm, BatchNorm
from torch_geometric.utils import add_self_loops, to_undirected

import torch.nn as nn
from torch.nn import Sequential as Seq
import torch.nn.functional as F

# class EdgeMLP(nn.Module):
#     def __init__(self, dims):
#         super().__init__()

#         self.act = nn.ReLU() 
#         self.layer1 = torch.nn.Linear(1, dims)
#         self.layer2 = torch.nn.Linear(dims, 1)

#         self.norm = nn.BatchNorm1d(dims, track_running_stats=False)

#     def forward(self, x):
#         if self.training == False:
#             self.eval()
#             self.norm.eval()
#         else:
#             self.train()
#             self.norm.train()

#         x = self.layer1(x)
#         x = self.act(x)
#         x = self.norm(x)
#         x = self.layer2(x)
#         return self.act(x)


class PoxelGCN(torch.nn.Module):
    def __init__(self, hidden_dim, out_dim, **kwargs):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=0.25)

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=3, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=3, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=3, concat=False)
        self.conv4 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=3, concat=False)
        self.conv5 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=3, concat=False)
        self.conv6 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=3, concat=False)

        self.pool1 = TopKPooling(hidden_dim, min_score=0.6)
        self.pool2 = TopKPooling(hidden_dim, min_score=0.5)
        self.pool3 = TopKPooling(hidden_dim, min_score=0.4)

        self.act = torch.nn.SiLU()
        edge_mlp_dim = 32 

        self.edge_MLP0 = Seq(nn.Linear(1, edge_mlp_dim), 
                             nn.ReLU(),
                            #  nn.BatchNorm1d(edge_mlp_dim),
                             nn.Linear(edge_mlp_dim, 1),
                             nn.ReLU())

        self.edge_MLP1 = Seq(nn.Linear(1, edge_mlp_dim), 
                             nn.ReLU(),
                            #  nn.BatchNorm1d(edge_mlp_dim),
                             nn.Linear(edge_mlp_dim, 1),
                             nn.ReLU())

        self.edge_MLP2 = Seq(nn.Linear(1, edge_mlp_dim), 
                             nn.ReLU(),
                            #  nn.BatchNorm1d(edge_mlp_dim),
                             nn.Linear(edge_mlp_dim, 1),
                             nn.ReLU())

        self.edge_MLP3 = Seq(nn.Linear(1, edge_mlp_dim), 
                             nn.ReLU(),
                            #  nn.BatchNorm1d(edge_mlp_dim),
                             nn.Linear(edge_mlp_dim, 1),
                             nn.ReLU())

        self.norm0 = BatchNorm(hidden_dim)
        self.norm1 = BatchNorm(hidden_dim)
        self.norm2 = BatchNorm(hidden_dim)
        self.norm3 = BatchNorm(hidden_dim)
        self.norm4 = BatchNorm(hidden_dim)
        self.norm5 = BatchNorm(hidden_dim)
        self.norm6 = BatchNorm(hidden_dim)


        self.readout_pool1 = AttentionalAggregation(
            MLP([hidden_dim, 1], act="relu"), 
            MLP([hidden_dim, hidden_dim*2], act="relu")
        )

        self.readout_pool2 = AttentionalAggregation(
            MLP([hidden_dim, 1], act="relu"), 
            MLP([hidden_dim, hidden_dim*2], act="relu")
        )

        self.global_pool = AttentionalAggregation(
            MLP([hidden_dim, 1], act="relu"), 
            MLP([hidden_dim, hidden_dim*2], act="relu")
        )

        self.mlp_out = torch.nn.Sequential(
            Lin(hidden_dim*6, out_dim),
            torch.nn.ReLU(),
            Lin(out_dim, out_dim)
            )

    def forward(self, data):
        x, pos, batch = (
            data.x,
            data.pos,
            data.batch,
        )
                
        x = self.norm0(x)

        edge_index = add_self_loops(to_undirected(knn_graph(pos, k=6, batch=batch, loop=True)))[0]
        edge_weights = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(dim=1)
        edge_weights = self.edge_MLP0(edge_weights).squeeze()
        
        h, _, _, batch, k_index, _ = self.pool1(x, edge_index, batch=batch)
        x = x[k_index]
        pos = pos[k_index]

        readout1 = self.readout_pool1(h, index=batch)
        
        edge_index = add_self_loops(to_undirected(knn_graph(pos, k=10, batch=batch, loop=True)))[0]
        edge_weights = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(dim=1)
        edge_weights = self.edge_MLP1(edge_weights).squeeze()

        h = self.act(h + self.conv1(h, edge_index, edge_weights))
        h = self.norm1(h)

        h = self.act(h + self.conv2(h, edge_index, edge_weights))
        h = self.norm2(h)


        h, _, _, batch, k_index, _ = self.pool2(h, edge_index, edge_attr=edge_weights, batch=batch)
        x = x[k_index]
        pos = pos[k_index]
        readout2 = self.readout_pool2(h, index=batch)
        
        edge_index = add_self_loops(to_undirected(knn_graph(pos, k=10, batch=batch, loop=True)))[0]
        edge_weights = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).unsqueeze(dim=1)
        edge_weights = self.edge_MLP2(edge_weights).squeeze()


        h = self.act(h + self.conv4(h, edge_index, edge_weights))
        h = self.norm4(h)


        h = self.act(h + self.conv5(h, edge_index, edge_weights))
        h = self.norm5(h)

        readout3 = self.global_pool(h, index=batch)

        h = torch.hstack([readout3,
                          readout2,
                          readout1])

        return self.mlp_out(h)