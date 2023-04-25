import torch
import torch_geometric
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import SAGPooling
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F


class SimplePoxelGCN(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()

        self.conv1 = GCN2Conv(hidden, 0.2)
        self.conv2 = GCN2Conv(hidden, 0.2)
        self.conv3 = GCN2Conv(hidden, 0.4)

        gate_nn = MLP([hidden, 1], act="relu")
        nn = MLP([hidden, hidden], act="relu")
        self.global_pool = AttentionalAggregation(gate_nn, nn)

    def forward(self, data):
        x, edge_index, edge_weights, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        h = self.conv1(x, x, edge_index, edge_weights)
        h = F.relu(h)

        h = self.conv2(h, x, edge_index, edge_weights)
        h = F.relu(h)

        h = self.conv3(h, x, edge_index, edge_weights)
        h = F.relu(h)

        h = self.global_pool(h, index=batch)
        return h
