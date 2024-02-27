import os
import re
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm

from genotype_autosomal import get_csv_list, load_csv


class Loader:
    def __init__(self, csv_path, build_id):
        self.csv_path = csv_path
        self.build_id = build_id

    def __call__(self, user_id):
        return load_csv(self.csv_path, f"user{user_id}_build{self.build_id}_autosomal.csv")

def get_csv_summary(csv_path, user_ids, build_id, n_proc=1):
    """Compute the sum of occurrences for each allele in the common RSIDs for the genomic data.
    Args:
        csv_path (str): the directory of the processed data.
        user_ids (list): the list of users to parse.
        build_id (int): the human reference assembly build.
    Returns:
        (DataFrama, set): The dataframe of sums, and the set of common RSIDs.
    """

    if n_proc > 1:
        with mp.Pool(n_proc) as p:
            dataframes = p.map(Loader(csv_path, build_id), user_ids)
    else:
        dataframes = []
        for user_id in user_ids:
            dataframes.append(load_csv(csv_path, f"user{user_id}_build{build_id}_autosomal.csv"))

    common_rsids = set.intersection(*map(lambda dataframe: set(dataframe.index), dataframes))
    summary_dataframe = pd.concat(dataframes)
    summary_dataframe = summary_dataframe.groupby(summary_dataframe.index).sum()
    summary_dataframe = summary_dataframe.loc[sorted(list(common_rsids))]
    return summary_dataframe, common_rsids


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
    print("num rsids missing", len(rsids_missing))
