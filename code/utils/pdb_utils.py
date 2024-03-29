from copy import deepcopy   
import torch
from torch_geometric.data import Data

def stringify_atom_idx(number, total_width):
        number = str(number)
        padding = total_width - len(number) 
        return " "*padding + number

def merge_pdbs(pdb_file_1, pdb_file_2):
    merged_pdb_content = ""
    atom_idx = -1
    
    with open(pdb_file_1, 'r') as pdb_in:
        for line in pdb_in:
            if line[:6].strip() in ['HETATM', 'ATOM']:
                atom_idx = int(line[6:11].strip())

            if line[:3] == 'END':
                continue

            merged_pdb_content += line 

    with open(pdb_file_2, 'r') as pdb_in:
        for line in pdb_in:
            if line[:6].strip() not in ['HETATM', 'ATOM']: 
                continue
            atom_idx += 1
            line = line[:6] + stringify_atom_idx(atom_idx, 5) + line[11:] 
            merged_pdb_content += line

    merged_pdb_content += "END"

    return merged_pdb_content

def get_pdb_atom_list(pdb_file: str, ligand: bool=False, deprotonate: bool=False) -> list:
    atom_list = []
    atom_type = "ATOM"

    if ligand:
        atom_type = "HETATM"

    line_ptrs = []

    # BETA FACTOR
    if not ligand:
        line_ptrs.append((60,66))

    # X, Y, and Z coord values
    line_ptrs.extend([(30,38), (38,46), (46,54)])

    with open(pdb_file, 'r') as pdb_in:  
        for line in pdb_in:
            if line[:6].strip() != atom_type:
                continue

            if deprotonate:
                if line[76:78].strip() == 'H':
                    continue
            
            atom_data = [line[12:16].strip()]
            atom_data.extend([float(line[ptr[0]:ptr[1]].strip()) for ptr in line_ptrs])
            atom_list.append(atom_data)

    return atom_list 

def pdb_to_graph(pdb_file: str, labels: list):
    graph_x = []
    graph_pos = []
    graph_beta_factor = []
    node_onehot = [0 for _ in labels]

    xyz_ptrs = ([(30,38), (38,46), (46,54)])

    with open(pdb_file, 'r') as pdb_in:
        for line in pdb_in:
            if line[:4] == 'ATOM':
                if line[76:78].strip() == 'H':
                    continue
                
                atom_label = line[12:16].strip()

                if atom_label not in labels:
                    continue

                node_label_onehot = deepcopy(node_onehot)
                node_label_onehot[labels.index(atom_label)] = 1
                graph_x.append(node_label_onehot)

                graph_beta_factor.append(float(line[60:66].strip()))

                graph_pos.append([float(line[ptr[0]:ptr[1]].strip()) for ptr in xyz_ptrs])

    return Data(x=torch.tensor(graph_x, dtype=torch.float),
                pos=torch.tensor(graph_pos, dtype=torch.float),
                beta=torch.tensor(graph_beta_factor, dtype=torch.float))

