import torch


def handle_data(batch):
    beta = batch.beta / 100
    x = torch.hstack((batch.x, beta.unsqueeze(1)))
    edge_attr = batch.edge_attr.unsqueeze(1) / 12
    return x, batch.edge_index, edge_attr
