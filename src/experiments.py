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
    csv_dataframe = get_csv_list('../data/opensnp/genotype/csv')

    # filter by human reference assembly build
    csv_dataframe = csv_dataframe[csv_dataframe['build_id'] == 37]

    # count up allele occurences, filter by RSIDs found throughout all user data
    csv_summary, common_rsids = get_csv_summary('../data/opensnp/genotype/csv', [1, 6], 36, n_proc=2)
    csv_summary.to_csv('../data/opensnp/genotype/sum/csv_summary.csv')

    print(len(common_rsids))
    print(csv_summary.head(20))
