import sys
import random
from glob import glob
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data
import torch
import pickle
import os

def consolidate_files(config: dict, in_dir: str, out_dir: str, partitions: int=100, **kwargs) -> None:
    if in_dir[-1] != '/':
        in_dir += '/'

    if out_dir[-1] != '/':
        out_dir += '/'

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    all_files = glob(in_dir + "*.pkl")
    file_template = out_dir + all_files[0].split('_')[-1].replace('.','s_%s.')

    file_indices = np.arange(len(all_files))
    np.random.shuffle(file_indices)
    index_groups = np.array_split(file_indices, partitions)

    for partition_idx, row in enumerate(index_groups):
        file_collection = []

        for file_idx in row: 
            small_file = all_files[file_idx]
            file_content = pickle.load(open(all_files[file_idx], 'rb'))
            file_collection.append(file_content)

        print(file_template % (partition_idx+1))
        pickle.dump(file_collection, open(file_template % (partition_idx+1), 'wb'))
        del file_collection
