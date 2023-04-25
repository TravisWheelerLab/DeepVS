import sys
import numpy as np
import os
import yaml
import argparse
from glob import glob

# from code.training_data_processing.get_plip_data import get_plip_data
from code.training_data_processing.generate_mol_graphs import generate_mol_graphs
from code.training_data_processing.generate_pocket_graphs import generate_pocket_graphs
from code.training_data_processing.generate_training_samples import (
    generate_training_samples,
)
from code.training_data_processing.consolidate_files import consolidate_files
from code.training_data_processing.generate_voxel_embeds import generate_voxel_embeds

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

args = parser.parse_args()

root = os.path.dirname(__file__) + "/"
CONFIG_PATH = root + "config.yaml"

with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

with open(args.params, "r") as config_file:
    params = yaml.safe_load(config_file)

params["config"] = config
params["skip"] = args.skip

action_dict = {
    # 'get_plip_data': data_processing.get_plip_data,
    "generate_mol_graphs": generate_mol_graphs,
    "generate_pocket_graphs": generate_pocket_graphs,
    "generate_training_samples": generate_training_samples,
    "consolidate_files": consolidate_files,
    "generate_voxel_embeds": generate_voxel_embeds,
}

holdout_pdbs = []

if args.holdout is not None:
    with open(args.holdout, "r") as txt_in:
        for line in txt_in:
            holdout_pdbs.append(line.strip())
    params["holdouts"] = holdout_pdbs

data_processing = True

if data_processing:
    pdb_ids = list(map(lambda x: x.split("/")[-2], glob(params["pdbbind_dir"] + "*/")))
    pdb_ids.remove("index")
    pdb_ids.remove("readme")
    pdb_ids = np.array(sorted(pdb_ids))
    id_batch = np.array_split(pdb_ids, args.batch_count)[args.batch_number - 1].tolist()
    params["id_batch"] = id_batch

if args.in_dir:
    params["in_dir"] = args.in_dir

if args.out_dir:
    params["out_dir"] = args.out_dir

action_dict[args.action](**params)
