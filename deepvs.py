import sys
import numpy as np
import os
import yaml
import argparse
from glob import glob

# from code.training_data_processing.get_plip_data import get_plip_data
from code.training_data_processing.generate_mol_graphs import generate_mol_graphs
from code.training_data_processing.generate_pocket_graphs import generate_pocket_graphs
from code.training_data_processing.generate_training_samples import generate_training_samples
from code.training_data_processing.consolidate_files import consolidate_files
from code.training_data_processing.generate_voxel_embeds import generate_voxel_embeds
from code.training_data_processing.generate_full_pockets import generate_full_pockets 
from code.training_data_processing.generate_ligand_smiles import generate_ligand_smiles 

from code.app.generate_pocket_embeds import generate_pocket_embeds 
from code.app.generate_mol_embeds import generate_mol_embeds 
from code.app.score_embeds import score_embeds 

parser = argparse.ArgumentParser(
    prog="DeepVS",
    description="Virtual screening for fun and profit.",
    epilog="For those in search of drugs.",
)

parser.add_argument("action", type=str)
parser.add_argument("params", type=str)
parser.add_argument("-n", "--batch_number", type=int, default=1)
parser.add_argument("-c", "--batch_count", type=int, default=1)
parser.add_argument("-s", "--skip", action="store_true")
parser.add_argument("--holdout", type=str)
parser.add_argument("--in_dir", type=str)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--pocket_file_in", type=str)
parser.add_argument("--mol_file_in", type=str)
parser.add_argument("--resolution", type=float)

parser.add_argument('--in_file', type=str)
parser.add_argument('--out_file', type=str)
parser.add_argument('--pocket_bounds_file', type=str)

args = parser.parse_args()

root = os.path.dirname(__file__) + "/"
CONFIG_PATH = root + "config.yaml"

with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

config['root_dir'] = os.path.dirname(os.path.realpath(__file__))

with open(args.params, "r") as param_file:
    params = yaml.safe_load(param_file)

params["skip"] = args.skip
app_actions = ['generate_pocket_embeds']

action_dict = {
    # 'get_plip_data': get_plip_data,
    "generate_mol_graphs": generate_mol_graphs,
    "generate_ligand_smiles": generate_ligand_smiles,
    "generate_pocket_graphs": generate_pocket_graphs,
    "generate_full_pockets": generate_full_pockets,
    "generate_training_samples": generate_training_samples,
    "generate_voxel_embeds": generate_voxel_embeds,
    "generate_pocket_embeds": generate_pocket_embeds,
    "generate_mol_embeds": generate_mol_embeds,
    "score_embeds": score_embeds,
}

holdout_pdbs = []

if args.holdout is not None:
    with open(args.holdout, "r") as txt_in:
        for line in txt_in:
            holdout_pdbs.append(line.strip())
    params["holdouts"] = holdout_pdbs

data_processing = True

if args.action in app_actions:
    data_processing = False

if data_processing:
    pdb_ids = list(map(lambda x: x.split("/")[-2], glob(params["pdbbind_dir"] + "*/")))

    if 'index' in pdb_ids:
        pdb_ids.remove("index")
    if 'readme' in pdb_ids:
        pdb_ids.remove("readme")

    pdb_ids = np.array(sorted(pdb_ids))
    id_batch = np.array_split(pdb_ids, args.batch_count)[args.batch_number - 1].tolist()
    params["id_batch"] = id_batch

if args.in_dir:
    params["in_dir"] = args.in_dir

if args.out_dir:
    params["out_dir"] = args.out_dir

if args.in_file:
    params['in_file'] = args.in_file

if args.pocket_file_in:
    params['pocket_file_in'] = args.pocket_file_in

if args.mol_file_in:
    params['mol_file_in'] = args.mol_file_in

if args.out_file:
    params['out_file'] = args.out_file

if args.pocket_bounds_file:
    params['pocket_bounds_file'] = args.pocket_bounds_file

if args.resolution:
    params['resolution'] = args.resolution

action_dict[args.action](**{**config, **params})
