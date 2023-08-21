import torch
import torch_geometric
from torch_geometric.nn import GCN2Conv, LayerNorm, global_mean_pool, InstanceNorm, GraphNorm, GATv2Conv
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import Sequential, Linear as Lin, ReLU, BatchNorm1d, Linear
from torch_geometric.nn.models import MLP 

import torch.nn.functional as F

class MolGCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim, **kwargs):
        super().__init__()

        self.linear1 = torch.nn.Linear(feature_dim, hidden_dim)

        # self.conv1 = GCN2Conv(hidden_dim, 0.2)
        # self.conv2 = GCN2Conv(hidden_dim, 0.2)
        # self.conv3 = GCN2Conv(hidden_dim, 0.2)
        # self.conv4 = GCN2Conv(hidden_dim, 0.2)

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=4, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=4, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=4, concat=False)
        self.conv4 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1, heads=4, concat=False)

        self.dropout = torch.nn.Dropout(p=0.25)
        self.batchnorm = torch_geometric.nn.BatchNorm(hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.norm3 = GraphNorm(hidden_dim)
        self.norm4 = GraphNorm(hidden_dim)
        self.norm5 = GraphNorm(hidden_dim)

        self.prediction_layer = Sequential(BatchNorm1d(hidden_dim),
                                        #  Linear(hidden_dim, hidden_dim),
                                        #  ReLU(),
                                        #  BatchNorm1d(hidden_dim),
                                         Linear(hidden_dim, out_dim))

        self.embed_layer = Sequential(Linear(hidden_dim, hidden_dim),
                                      ReLU(),
                                      BatchNorm1d(hidden_dim),
                                      Linear(hidden_dim, hidden_dim))

        self.act = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.linear1(x)

        x = self.dropout(x)
        h = self.act(x + self.conv1(x, edge_index))
        h = self.norm1(h)

        h = self.dropout(h)
        h = self.act(h + self.conv2(h, edge_index))
        h = self.norm2(h)

        h = self.dropout(h)
        h = self.act(h + self.conv3(h, edge_index))
        h = self.norm3(h)

        # h = self.act(h + self.conv4(h, edge_index))
        # h = self.norm4(h)

        h = self.embed_layer(h)
        atom_preds = self.prediction_layer(h)

        return h, atom_preds 
        # return  #normalize(x, dim=-1, p=2), self.atom_classifier(x), self.lin2(out)