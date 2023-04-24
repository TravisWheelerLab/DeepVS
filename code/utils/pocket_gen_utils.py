import numpy as np
import code.utils.pdb_utils as pdb_utils
from code.utils.EdgeData import EdgeData
import torch
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data


def get_box_from_ligand_diagonal(ligand_atom_coords, padding, resolution):
    ligand_atom_coords = np.array(ligand_atom_coords)
    voxel_coords = []
    padding_array = np.array([padding] * 3)

    max_xyz = (
        np.array(
            [
                np.max(ligand_atom_coords[:, 0]),
                np.max(ligand_atom_coords[:, 1]),
                np.max(ligand_atom_coords[:, 2]),
            ]
        )
        + padding_array
    )

    min_xyz = (
        np.array(
            [
                np.min(ligand_atom_coords[:, 0]),
                np.min(ligand_atom_coords[:, 1]),
                np.min(ligand_atom_coords[:, 2]),
            ]
        )
        + padding_array
    )

    for x_val in np.arange(min_xyz[0], max_xyz[0], resolution):
        for y_val in np.arange(min_xyz[1], max_xyz[1], resolution):
            for z_val in np.arange(min_xyz[2], max_xyz[2], resolution):
                voxel_coords.append([x_val, y_val, z_val])

    return voxel_coords


def get_voxel_coords(
    pdb_file: str,
    resolution: float = 1.0,
    ligand_atom_coords: list = None,
    min_coord: list = None,
    max_coord: list = None,
    padding: float = 1,
    shrinkwrap: bool = False,
) -> list:
    if ligand_atom_coords:
        return get_box_from_ligand_diagonal(ligand_atom_coords, padding, resolution)


def generate_voxel_graphs(
    voxel_indices: list,
    pocket_graph: object,
    pdb_id: str,
    neighbor_count: int,
    EDGE_LABELS: list,
):
    pos_masked = deepcopy(pocket_graph.pos)
    pos_masked[voxel_indices] = torch.tensor([float("inf")] * 3)

    graph_collection = []

    for voxel_index in voxel_indices:
        edge_data = EdgeData(EDGE_LABELS)

        neighbor_atom_indices = torch.sort(
            torch.cdist(pocket_graph.pos[voxel_index].unsqueeze(0), pos_masked)[0]
        )[1][:neighbor_count]

        voxel_graph_indices = torch.hstack((voxel_index, neighbor_atom_indices))

        for n_i in range(voxel_graph_indices.size(0)):
            node_distances = torch.cdist(
                pocket_graph.pos[voxel_graph_indices[n_i]].unsqueeze(0),
                pocket_graph.pos[voxel_graph_indices[n_i:]],
            )[0]

            for n_j, d in enumerate(node_distances):
                if n_j == 0:
                    if n_i == 0:
                        edge_label = "voxel-self"
                    else:
                        edge_label = "atom-self"
                else:
                    if n_i == 0:
                        edge_label = "atom-voxel"
                    else:
                        edge_label = "atom-atom"

                edge_data.add_edge(n_i, n_i + n_j, d.item(), edge_label)

        g_edge_index, g_edge_attr, g_edge_labels = edge_data.get_data()

        vox_graph = Data(
            x=pocket_graph.x[voxel_graph_indices],
            y=pocket_graph.y[voxel_graph_indices[0]].unsqueeze(0),
            pos=pocket_graph.pos[voxel_graph_indices],
            contact_map=pocket_graph.contact_map[voxel_graph_indices[0]],
            beta=pocket_graph.beta[voxel_graph_indices],
            pdb_id=pdb_id,
            edge_index=torch.tensor(g_edge_index, dtype=torch.long),
            edge_attr=torch.tensor(g_edge_attr, dtype=torch.float),
            edge_labels=torch.tensor(g_edge_labels, dtype=torch.long),
        )

        graph_collection.append(vox_graph)

    return graph_collection


def get_pocket_graph(
    config,
    pdb_file: str,
    voxel_coords: list,
    interaction_profile: list = None,
    mol_graph=None,
    neighbor_count: int = 10,
):
    POCKET_ATOM_LABELS = config["POCKET_ATOM_LABELS"]
    INTERACTION_LABELS = config["INTERACTION_LABELS"]

    protein_atom_data = pdb_utils.get_pdb_atom_list(pdb_file, deprotonate=True)
    protein_atom_coords = torch.tensor(
        [x[-3:] for x in protein_atom_data], requires_grad=False
    ).double()

    pocket_graph_nodes = [["VOXEL", 0] + x for x in voxel_coords]

    voxel_coords = torch.tensor(voxel_coords, requires_grad=False).double()

    # For every voxel point, get N closest protein atoms and add them to graph node list
    neighbor_atom_indices = (
        torch.sort(torch.cdist(voxel_coords, protein_atom_coords, p=2), dim=1)[1][
            :, :neighbor_count
        ]
        .flatten()
        .unique()
    )

    for atom_index in neighbor_atom_indices:
        pocket_graph_nodes.append(protein_atom_data[atom_index.item()])

    graph_x = []
    graph_pos = []
    graph_beta_factor = []
    node_onehot = [0 for _ in POCKET_ATOM_LABELS]

    for node in pocket_graph_nodes:
        node_label_onehot = deepcopy(node_onehot)
        node_label_onehot[POCKET_ATOM_LABELS.index(node[0])] = 1
        graph_beta_factor.append(node[1])
        graph_x.append(node_label_onehot)
        graph_pos.append(node[-3:])

    pocket_graph_y = np.zeros((np.size(pocket_graph_nodes, 0), len(INTERACTION_LABELS)))
    contact_map = [-1 for x in pocket_graph_y]

    if mol_graph:
        # For every atom in ligand, properly label closest voxel point
        neighbor_voxel_indices = torch.sort(
            torch.cdist(mol_graph.pos.double(), voxel_coords)
        )[1][:, 0]

        for node_index, voxel_index in enumerate(neighbor_voxel_indices):
            if mol_graph.heavy[node_index] == 1:
                contact_map[voxel_index] = node_index

    if interaction_profile:
        # For every interaction in interaction profile, properly label closest voxel point
        ip_coords = torch.tensor([x[1] for x in interaction_profile]).double()
        neighbor_voxel_indices = torch.sort(torch.cdist(ip_coords, voxel_coords))[1][
            :, 0
        ]

        for ip_index, voxel_index in enumerate(neighbor_voxel_indices):
            interaction_index = INTERACTION_LABELS.index(
                interaction_profile[ip_index][0]
            )
            pocket_graph_y[voxel_index][interaction_index] = 1

    pocket_graph = Data(
        x=torch.tensor(graph_x, dtype=torch.float),
        y=torch.tensor(pocket_graph_y, dtype=torch.long),
        pos=torch.tensor(graph_pos, dtype=torch.float),
        beta=torch.tensor(graph_beta_factor, dtype=torch.float),
        contact_map=torch.tensor(contact_map, dtype=torch.long),
    )

    return pocket_graph
