from importlib import import_module
import importlib.util
import sys
from copy import deepcopy
import numpy as np
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from code.utils.EdgeData import EdgeData
import torch
import pickle
import os
import code.utils.data_processing_utils as data_utils
import torch
import torch_geometric
import os
import code.utils.pocket_gen_utils as pocket_gen_utils
from torch_geometric.loader import DataLoader


def generate_voxel_embeds(
    id_batch: list,
    vox_embedder_model: str,
    vox_embedder_weights: str,
    vox_embedder_hyperparams: dict,
    neighbor_count: int = 10,
    pocket_graph_dir: str = None,
    vox_embed_graph_dir: str = None,
    data_dir: str = None,
    **kwargs
) -> None:
    pocket_graph_dir, pocket_graph_ft = data_utils.get_output_paths(
        pocket_graph_dir, data_dir, kwargs["pocket_graph_file_template"]
    )
    vox_embed_graph_dir, vox_embed_graph_ft = data_utils.get_output_paths(
        vox_embed_graph_dir, data_dir, kwargs["vox_embed_graph_file_template"]
    )

    vox_embedder_model = data_utils.interpolate_root(vox_embedder_model, kwargs['root_dir'])
    vox_embedder_weights = data_utils.interpolate_root(vox_embedder_weights, kwargs['root_dir'])

    POCKET_ATOM_LABELS = kwargs["POCKET_ATOM_LABELS"]
    voxel_label_index = POCKET_ATOM_LABELS.index("VOXEL")

    if os.path.exists(vox_embed_graph_dir) == False:
        os.makedirs(vox_embed_graph_dir)

    if kwargs.get("skip"):
        id_batch = data_utils.trim_batch_ids(id_batch, vox_embed_graph_ft)

    batch_total = len(id_batch)

    VoxEmbedder = data_utils.load_class_from_file(vox_embedder_model)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    vox_embedder = VoxEmbedder(
        **vox_embedder_hyperparams,
        # ).to(device)
    )

    vox_embedder.load_state_dict(torch.load(vox_embedder_weights, map_location=device))
    vox_embedder.eval()

    for pdb_i, pdb_id in enumerate(id_batch):
        print("!%s: %s of %s" % (pdb_id, pdb_i + 1, batch_total))
        pocket_file = pocket_graph_ft % pdb_id
        vox_embed_graph_file = vox_embed_graph_ft % pdb_id

        if os.path.exists(pocket_file) == False:
            continue

        pocket_graph = pickle.load(open(pocket_file, "rb"))
        voxel_indices = torch.where(pocket_graph.x[:, voxel_label_index] == 1)[0]

        voxel_graphs = pocket_gen_utils.generate_voxel_graphs(
            voxel_indices, pocket_graph, pdb_id, neighbor_count, EDGE_LABELS
        )

        voxel_graphs = next(
            iter(DataLoader(voxel_graphs, batch_size=len(voxel_graphs), shuffle=False))
        )

        voxel_node_indices = torch.where(
            voxel_graphs.x[:, POCKET_ATOM_LABELS.index("VOXEL")] == 1
        )

        voxel_graphs = voxel_graphs.to(device)

        with torch.no_grad():
            out, _ = vox_embedder(voxel_graphs)

        out = out.cpu()
        voxel_graphs = voxel_graphs.cpu()

        poxel_pos = voxel_graphs.pos[voxel_node_indices]
        poxel_x = out[voxel_node_indices]

        edge_data = EdgeData(["adjacent", "self"], pocket_graph.resolution)

        # for n_i in range(poxel_x.size(0)):
        #     node_distances = torch.cdist(
        #         poxel_pos[n_i].unsqueeze(0),
        #         poxel_pos[n_i:],
        #     )[0]

        #     for n_j, d in enumerate(node_distances):
        #         edge_label = "adjacent"

        #         if n_j == 0:
        #             edge_label = "self"

        #         edge_data.add_edge(n_i, n_i + n_j, d.item(), edge_label)

        # g_edge_index, g_edge_attr, g_edge_labels = edge_data.get_data()

        poxel_graph = Data(
            x=poxel_x,
            pos=poxel_pos,
            pdb_id=pdb_id,
            # edge_index=g_edge_index,
            # edge_attr=g_edge_attr,
            # edge_labels=g_edge_labels,
        )

        pickle.dump(poxel_graph, open(vox_embed_graph_file, "wb"))
