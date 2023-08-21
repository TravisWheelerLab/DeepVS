import torch
import torch_geometric
from torch_geometric.nn import GCN2Conv, LayerNorm, global_mean_pool, InstanceNorm, GraphConv, GraphNorm, GATv2Conv, BatchNorm
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn.pool import ASAPooling, TopKPooling

from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation
from torch_geometric.nn.models import MLP 

import torch.nn.functional as F

class MolAggregator(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()

        mlp_h = MLP([hidden_dim, 1])
        mlp_theta = MLP([hidden_dim, hidden_dim])

        # self.conv1 = GCN2Conv(hidden_dim, 0.2)
        # self.conv2 = GCN2Conv(hidden_dim, 0.2)
        # self.conv3 = GCN2Conv(hidden_dim, 0.2)
        # self.conv4 = GCN2Conv(hidden_dim, 0.2)

        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=3, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=3, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=3, concat=False)
        self.conv4 = GATv2Conv(hidden_dim, hidden_dim, heads=3, concat=False)
        self.conv5 = GATv2Conv(hidden_dim, hidden_dim, heads=3, concat=False)
        self.conv6 = GATv2Conv(hidden_dim, hidden_dim, heads=3, concat=False)

        self.pool1 = TopKPooling(hidden_dim, min_score=0.6)
        self.pool2 = TopKPooling(hidden_dim, min_score=0.5)

        self.dropout = torch.nn.Dropout(p=0.25)
        self.batchnorm = torch_geometric.nn.BatchNorm(hidden_dim)

        self.norm0 = BatchNorm(hidden_dim)
        self.norm1 = BatchNorm(hidden_dim)
        self.norm2 = BatchNorm(hidden_dim)
        self.norm3 = BatchNorm(hidden_dim)
        self.norm4 = BatchNorm(hidden_dim)
        self.norm5 = BatchNorm(hidden_dim)
        self.norm6 = BatchNorm(hidden_dim)

        self.act = torch.nn.ReLU()

        self.mol_aggr = AttentionalAggregation(MLP([hidden_dim, 1], act="relu"), 
                                               MLP([hidden_dim, hidden_dim*2], act="relu"))

        self.readout_pool1 = AttentionalAggregation(MLP([hidden_dim, 1], act="relu"), 
                                                    MLP([hidden_dim, hidden_dim*2], act="relu"))

        self.readout_pool2 = AttentionalAggregation(MLP([hidden_dim, 1], act="relu"), 
                                                    MLP([hidden_dim, hidden_dim*2], act="relu"))

        self.mlp_out = Seq(
            Lin(hidden_dim*6, out_dim),
            ReLU(),
            Lin(out_dim, out_dim)
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.norm0(x)

        readout1 = self.readout_pool1(x, index=batch)
        h, edge_index, _, batch, k_index, _ = self.pool1(x, edge_index, batch=batch)

        # h = self.dropout(h)
        h = self.act(h + self.conv1(h, edge_index))
        h = self.norm1(h)

        # h = self.dropout(h)
        h = self.act(h + self.conv2(h, edge_index))
        h = self.norm2(h)
        
        # h = self.act(h + self.conv3(h, edge_index))
        # h = self.norm3(h)

        readout2 = self.readout_pool2(h, index=batch)
        h, edge_index, _, batch, k_index, _ = self.pool2(h, edge_index, batch=batch)

        # h = self.dropout(h)
        h = self.act(h + self.conv4(h, edge_index))
        h = self.norm4(h)

        # h = self.dropout(h)
        h = self.act(h + self.conv5(h, edge_index))
        h = self.norm5(h)
        
        # h = self.act(h + self.conv6(h, edge_index))
        # h = self.norm6(h)

        h = torch.hstack([self.mol_aggr(h, index=batch),
                          readout2,
                          readout1])
        
        return self.mlp_out(h)