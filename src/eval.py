import multiprocessing as mp
import os
import re

import pandas as pd
from tqdm import tqdm

from src.genotypes.genotype import Genotype



# print(new_users - prev_users)

def compare_rsids(prev_path, new_path, diff_path, user_id, build):
    prev_file_path = os.path.join(prev_path, f'user{user_id}_build{build}_autosomal.csv')
    new_file_path = os.path.join(new_path, f'user{user_id}_build{build}.csv')
    prev_df = pd.read_csv(prev_file_path, index_col='rsid')
    new_df = pd.read_csv(new_file_path, index_col='rsid')
    diff_df = prev_df[~prev_df.index.isin(new_df.index)]
    if len(diff_df) > 0:
        os.makedirs(diff_path, exist_ok=True)
        diff_df.to_csv(os.path.join(diff_path, f'user{user_id}_build{build}_diff.csv'))


# user_6552_genotype = Genotype()
# user_6552_genotype.from_user_id('../data/genotype/raw', "../res", 2718, 37)
# user_6552_genotype.clean()
# user_6552_genotype.filter_rsids_proprietary()
# user_6552_genotype.filter_chromosomes_autosomal()
# user_6552_genotype.save('../data/genotype/out')
#
# compare_rsids(prev_path, new_path, diff_path, 2718, 37)

# for user_id in tqdm(new_users):
#     if user_id not in prev_users:
#         continue
#     compare_rsids(prev_path, new_path, diff_path, user_id, 37)


if __name__ == '__main__':
    prev_path = '../data/genotype/csv'
    new_path = '../data/genotype/out'
    diff_path = '../data/genotype/diff'

    prev_users = set()
    prev_files = os.listdir(prev_path)
    for prev_file in prev_files:
        match = re.match(r'user(\d+)_build(\d+)_autosomal.csv', prev_file)
        user_id = int(match.group(1))
        build = int(match.group(2))
        if build != 37:
            continue
        prev_users.add(user_id)

    new_users = set()
    new_files = os.listdir(new_path)
    for new_file in new_files:
        match = re.match(r'user(\d+)_build(\d+).csv', new_file)
        user_id = int(match.group(1))
        build = int(match.group(2))
        if build != 37:
            continue
        new_users.add(user_id)

    mp.freeze_support()

    with mp.Pool(os.cpu_count() - 1) as pool:
        for user_id in list(new_users & prev_users):
            pool.apply_async(compare_rsids, args=(prev_path, new_path, diff_path, user_id, 37))
        pool.close()
        pool.join()
