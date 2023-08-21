from copy import deepcopy
import os
import pickle
import sys
import numpy as np
import code.utils.data_processing_utils as data_utils
import code.utils.pdb_utils as pdb_utils
import code.utils.pocket_gen_utils as pocket_gen_utils
import code.utils.application_utils as app_utils 
from torch_geometric.nn import radius_graph 
from torch_geometric.nn import knn
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.loader import DataLoader


# from code.pdbbind_data_processing.get_voxel_coords import get_voxel_coords
# from code.utils.get_distance import get_distance

import torch
from torch_geometric.data import Data

def get_bounding_box(point_coordinates, resolution):
    min_xyz, _ = torch.min(point_coordinates, 0)
    max_xyz, _ = torch.max(point_coordinates, 0)

    min_xyz -= resolution*2
    max_xyz += resolution*2

    x_range = torch.arange(min_xyz[0], max_xyz[0], resolution)
    y_range = torch.arange(min_xyz[1], max_xyz[1], resolution)
    z_range = torch.arange(min_xyz[2], max_xyz[2], resolution)

    return torch.cartesian_prod(x_range,y_range,z_range)

# def get_vox_embed(pdb_graph, 
#                   vox_embed_model,
#                   voxel_coords, 
#                   neighbor_count, 
#                   labels,
#                   pdb_id: None):

#     poxel_x = None
#     poxel_pos = None

#     voxel_onehot = torch.zeros(len(labels))
#     voxel_onehot[labels.index('VOXEL')] = 1

#     voxel_atom_edges = knn(pdb_graph.pos, voxel_coords, neighbor_count) 

#     with torch.no_grad():
#         for voxel_idx in torch.unique(torch.sort(voxel_atom_edges[0])[0]):
#             nearest_atoms_mask = voxel_atom_edges[0]==voxel_idx
#             nearest_atoms = voxel_atom_edges[1][nearest_atoms_mask]

#             graph_x = torch.vstack((voxel_onehot, pdb_graph.x[nearest_atoms]))

#             graph_pos = torch.vstack((voxel_coords[voxel_idx], pdb_graph.pos[nearest_atoms]))
#             # graph_list.append(Data(x=graph_x, pos=graph_pos))

#             out, _ = vox_embed_model(Data(x=graph_x, pos=graph_pos))

#             if poxel_x is None:
#                 poxel_x = out[0]
#                 poxel_pos = voxel_coords[voxel_idx]
#             else:
#                 poxel_x = torch.vstack((poxel_x, out[0]))
#                 poxel_pos = torch.vstack((poxel_pos, voxel_coords[voxel_idx]))
    
#     return Data(x=poxel_x, pos=poxel_pos, pdb_id=pdb_id) 


def generate_voxel_embeds(
    id_batch: list,
    pdbbind_dir: str,
    mol_graph_dir: str = None,
    data_dir: str = None,
    voxel_embed_dir: str = None,
    resolution: float = 1.0,
    neighbor_count: int = 10,
    **kwargs
) -> None:
    mol_graph_dir, mol_graph_ft = data_utils.get_output_paths(
        mol_graph_dir, data_dir, kwargs["mol_graph_file_template"]
    )
    
    voxel_embed_dir, voxel_embed_ft = data_utils.get_output_paths(
        voxel_embed_dir, data_dir, kwargs["vox_embed_graph_file_template"]
    )

    vox_embedder = data_utils.interpolate_root(kwargs['vox_embedder_model'], kwargs['root_dir'])
    vox_embedder_weights = data_utils.interpolate_root(kwargs['vox_embedder_weights'], kwargs['root_dir'])

    VoxEmbedder = data_utils.load_class_from_file(vox_embedder)
    vox_embedder_model = VoxEmbedder(**kwargs['vox_embedder_hyperparams'])

    vox_embedder_model.load_state_dict(torch.load(vox_embedder_weights, map_location='cpu'))
    vox_embedder_model.eval()

    POCKET_ATOM_LABELS = kwargs['POCKET_ATOM_LABELS']

    if os.path.exists(voxel_embed_dir) == False:
        os.makedirs(voxel_embed_dir)

    pocket_pdb_ft = pdbbind_dir + "%s/%s_protein.pdb"

    if kwargs.get("skip"):
        id_batch = data_utils.trim_batch_ids(id_batch, voxel_embed_ft)

    batch_total = len(id_batch)

    for pdb_i, pdb_id in enumerate(id_batch):
        print("generating %s: %s of %s" % (pdb_id, pdb_i + 1, batch_total))

        mol_graph = mol_graph_ft % pdb_id
        pocket_pdb = pocket_pdb_ft % (pdb_id, pdb_id)
        voxel_embed_file = voxel_embed_ft % pdb_id

        skip = False

        for f in [mol_graph, pocket_pdb]:
            if os.path.exists(f) == False:
                skip = True

        if skip:
            continue

        pdb_graph = pdb_utils.pdb_to_graph(pocket_pdb, POCKET_ATOM_LABELS) 
        mol_graph = pickle.load(open(mol_graph, "rb"))
        vox_coords = get_bounding_box(mol_graph.pos, resolution)

        vox_embed_graph = app_utils.get_vox_embed(pdb_graph, 
                                                  vox_embedder_model,
                                                  vox_coords, 
                                                  neighbor_count, 
                                                  POCKET_ATOM_LABELS)

        vox_embed_graph.pdb_id = pdb_id
        print(voxel_embed_file)
        pickle.dump(vox_embed_graph, open(voxel_embed_file, "wb"))

