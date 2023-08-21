import torch
from torch_geometric.nn import knn
from torch_geometric.data import Data
from torch_geometric import utils as pyg_utils


def get_bounding_box(point_coordinates, resolution):
    min_xyz, _ = torch.min(point_coordinates, 0)
    max_xyz, _ = torch.max(point_coordinates, 0)

    min_xyz -= resolution*2
    max_xyz += resolution*2

    x_range = torch.arange(min_xyz[0], max_xyz[0], resolution)
    y_range = torch.arange(min_xyz[1], max_xyz[1], resolution)
    z_range = torch.arange(min_xyz[2], max_xyz[2], resolution)

    return torch.cartesian_prod(x_range,y_range,z_range)

def get_vox_embed(pdb_graph, 
                  vox_embed_model,
                  voxel_coords, 
                  neighbor_count, 
                  labels):

    poxel_x = None
    poxel_pos = None

    voxel_onehot = torch.zeros(len(labels))
    voxel_onehot[labels.index('VOXEL')] = 1

    voxel_atom_edges = knn(pdb_graph.pos, voxel_coords, neighbor_count) 

    with torch.no_grad():
        for voxel_idx in torch.unique(torch.sort(voxel_atom_edges[0])[0]):
            nearest_atoms_mask = voxel_atom_edges[0]==voxel_idx
            nearest_atoms = voxel_atom_edges[1][nearest_atoms_mask]

            graph_x = torch.vstack((voxel_onehot, pdb_graph.x[nearest_atoms]))

            graph_pos = torch.vstack((voxel_coords[voxel_idx], pdb_graph.pos[nearest_atoms]))
            # graph_list.append(Data(x=graph_x, pos=graph_pos))

            out, _ = vox_embed_model(Data(x=graph_x, pos=graph_pos))

            if poxel_x is None:
                poxel_x = out[0]
                poxel_pos = voxel_coords[voxel_idx]
            else:
                poxel_x = torch.vstack((poxel_x, out[0]))
                poxel_pos = torch.vstack((poxel_pos, voxel_coords[voxel_idx]))
    
    return Data(x=poxel_x, pos=poxel_pos) 

def graph_from_smile(smile_string):
    feature_list = [[5, 6, 7, 8, 9, 14, 15, 16, 17, 23, 26, 29, 33, 34, 35, 44, 51, 53, 77, 78],
                    [0, 1, 2],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [4, 5, 6, 7, 8],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1],
                    [0, 2, 3, 4, 5, 6],
                    [0, 1],
                    [0, 1]]
                
    edge_feature_list = [[1, 2, 3, 4],
                         [0, 1, 2],
                         [0, 1]]

    g = pyg_utils.from_smiles(smile_string, with_hydrogen=False)
    new_x = []
    new_edge_attr = []

    for row in g.x:
        new_row = []
        row = row.tolist()

        for feature_i, feature_val in enumerate(row):
            feature_domain = feature_list[feature_i]

            if len(feature_domain) <= 2:
                new_row.append(feature_val)
            else:
                onehot = [0 for _ in range(len(feature_domain))]
                onehot[feature_domain.index(feature_val)] = 1
                new_row += onehot

        new_x.append(new_row)

    for row in g.edge_attr:
        new_row = []
        row = row.tolist()

        for feature_i, feature_val in enumerate(row):
            feature_domain = edge_feature_list[feature_i]

            if len(feature_domain) <= 2:
                new_row.append(feature_val)
            else:
                onehot = [0 for _ in range(len(feature_domain))]
                onehot[feature_domain.index(feature_val)] = 1
                new_row += onehot

        new_edge_attr.append(new_row)
    
    g.x = torch.tensor(new_x)
    g.edge_attr = torch.tensor(new_edge_attr)
    return g