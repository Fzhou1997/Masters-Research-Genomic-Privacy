import os
import re
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm

from genotype_autosomal import get_csv_list, load_csv

import numpy as np


def get_npy_user_ids(npy_path, build):
    files_list = os.listdir(npy_path)
    user_ids = [int(re.search(r'user(\d+)_', file).group(1)) for file in files_list if int(re.search(r'build(\d+)_', file).group(1)) == build and file.startswith('user')]
    return user_ids


def load_npy(user_id, build=37):
    return np.load(f'../data/opensnp/genotype/npy/user{user_id}_build{build}_autosomal.npy')


if __name__ == '__main__':
    build = 37
    user_ids = sorted(get_npy_user_ids('../data/opensnp/genotype/npy', build))

    npy_vectors = [load_npy(user_id, build=build) for user_id in user_ids]
    G = np.vstack(npy_vectors)
    np.save(f'../data/opensnp/genotype/npy/G_build{build}_autosomal.npy', G)

    print(G.shape)
    print(G[:10, :10])

    print('Count of NaN values:', np.count_nonzero(G == -1))
    missing = np.argwhere(G == -1)
    print(missing.shape)
    users_missing, users_missing_counts = np.unique(missing[:, 0], return_counts=True)
    print("user missing", users_missing)
    print("num users missing", len(users_missing))
    rsids_missing, rsids_missing_counts = np.unique(missing[:, 1], return_counts=True)
    print("rsid missing", rsids_missing)
    print("num stat missing", len(rsids_missing))
