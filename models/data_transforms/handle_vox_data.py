import torch
import sys

def handle_vox_data(batch):
    norm_val = 15
    beta = batch.beta / 100
    x = torch.hstack((batch.x, beta.unsqueeze(1)))
    # max_weights = torch.zeros_like(batch.edge_attr)
    # max_weights[:] = norm_val
    # edge_attr = (max_weights-batch.edge_attr)/norm_val
    # edge_attr[edge_attr < 0] = 0
    return x, batch.pos, batch.batch
