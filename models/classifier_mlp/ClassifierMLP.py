import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class ResLayer(nn.Module):
    def __init__(self, dims, act=nn.ReLU, dr=0.25):
        super().__init__()
        
        self.act = act()
        self.layer = nn.Linear(dims, dims)

        self.dropout = nn.Dropout(dr)
        self.norm = nn.BatchNorm1d(dims)

    def forward(self, x):
        x = x + self.layer(x)
        x = self.act(x)
        x = self.norm(x)

        return self.dropout(x)

class ClassifierMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, res_layers=3):
        super().__init__()

        self.res_layers = []

        for _ in range(res_layers):
            self.res_layers.append(ResLayer(hidden_dim, dr=0.25))

        self.res_layers = nn.Sequential(*self.res_layers)

        # self.linear_in = nn.Linear(in_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.res_layers(x)
        x = self.linear_out(x)
        return x