import os
import re
import warnings
import multiprocessing as mp

import pandas as pd

from snps import SNPs

from phenotype_hair_color import load_processed


# region <csv processors>
def process_users_csvs(csv_path, build_id):
    """

    """
    csv_dataframe = get_csv_list(csv_path)
    csv_dataframe = csv_dataframe[csv_dataframe['build_id'] == build_id]
    user_ids = csv_dataframe['user_id'].unique()

    dataframes = []

    counter_dataframe = None
    for user_id in user_ids:
        if counter_dataframe is None:
            counter_dataframe = load_csv(csv_path, f"user{user_id}_build{build_id}_autosomal.csv")
        else:
            counter_dataframe = counter_dataframe.add(load_csv(csv_path, f"user{user_id}_build{build_id}_autosomal.csv"), fill_value=0)


def load_csv(csv_path, csv_name):
    dataframe = pd.read_csv(os.path.join(csv_path, csv_name), index_col=0)
    return dataframe


def get_csv_list(csv_path):
    file_list = os.listdir(csv_path)
    csv_list = []
    for file_name in file_list:
        file_extension = file_name.split('.')[-1]
        if file_extension != 'csv':
            continue
        match = re.search(r"user(\d+)_build(\d+)_autosomal\.csv", file_name)
        if match:
            user_id = int(match.group(1))
            build_id = int(match.group(2))
            csv_list.append([user_id, build_id])
    dataframe = pd.DataFrame(csv_list, columns=['user_id', 'build_id'])
    dataframe = dataframe.sort_values(by=['user_id', 'build_id'])
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def save_csv(csv_path, snps_dataframe, user_id, build_id):
    csv_name = f"user{user_id}_build{build_id}_autosomal.csv"
    file_path = os.path.join(csv_path, csv_name)
    csv_dataframe = snps_dataframe[snps_dataframe['chrom'].isin([str(i) for i in range(1, 23)])]
    csv_dataframe.index = csv_dataframe.index.str.lstrip('#')
    csv_dataframe = csv_dataframe[csv_dataframe.index.str.startswith('rs')]
    # rsid, A, C, G, T, I, D, -
    # str, int, int, int, int, int, int, int
    csv_dataframe['genotype'] = csv_dataframe['genotype'].fillna('--')
    csv_dataframe['a'] = csv_dataframe['genotype'].apply(lambda genotype: genotype.count('A'))
    csv_dataframe['c'] = csv_dataframe['genotype'].apply(lambda genotype: genotype.count('C'))
    csv_dataframe['g'] = csv_dataframe['genotype'].apply(lambda genotype: genotype.count('G'))
    csv_dataframe['t'] = csv_dataframe['genotype'].apply(lambda genotype: genotype.count('T'))
    csv_dataframe['i'] = csv_dataframe['genotype'].apply(lambda genotype: genotype.count('I'))
    csv_dataframe['d'] = csv_dataframe['genotype'].apply(lambda genotype: genotype.count('D'))
    csv_dataframe['-'] = csv_dataframe['genotype'].apply(lambda genotype: genotype.count('-'))
    csv_dataframe = csv_dataframe.drop(columns=['chrom', 'pos', 'genotype'])
    csv_dataframe.to_csv(file_path)
# endregion


# region <txt processors>
class Processor:
    def __init__(self, txt_path, txt_list, csv_path):
        self.txt_path = txt_path
        self.txt_list = txt_list
        self.csv_path = csv_path

    def __call__(self, user_id):
        process_user_txts(self.txt_path, self.txt_list, self.csv_path, user_id)


def process_users_txts(txt_path, csv_path, user_ids, n_proc=1):
    """Process the user-submitted genome files for a given list of user IDs.
    Args:
        txt_path (str):  path to directory containing txt files with genome data.
        csv_path (str):  path to directory to save processed csv files to.
        user_ids (list): list of user IDs to process.
        n_proc (int):    optional number of process to parallelize over.
    """
    txt_list = get_txt_list(txt_path, user_ids)
    if n_proc > 1:
        with mp.Pool(n_proc) as p:
            p.map(Processor(txt_path, txt_list, csv_path), user_ids)
    else:
        for user_id in user_ids:
            Processor(txt_path, txt_list, csv_path)(user_id)


def process_user_txts(txt_path, txt_list, csv_path, user_id):
    """Process the user-submitted genome files for a given user ID.
    Args:
        txt_path (str):       path to directory containing txt files with genome data.
        txt_list (DataFrame): DataFrame of filenames to read from the txt file directory.
        csv_path (str):       path to directory to save processed csv files to.
        user_id (int):        the user ID to process.
    """
    txt_list = txt_list[txt_list['user_id'] == user_id]
    txt_list = txt_list.sort_values(by=['file_id'])
    txt_list = txt_list.reset_index(drop=True)
    snps = {}
    for i in range(len(txt_list)):
        txt_row = txt_list.iloc[i]
        txt_name = f"user{txt_row['user_id']}_file{txt_row['file_id']}_yearofbirth_{txt_row['year_of_birth']}_sex_{txt_row['sex']}.{txt_row['provider']}.txt"
        try:
            current = load_txt(txt_path, txt_name)
        except Exception as e:
            print(f"Error loading {txt_name}: {e}")
            continue
        if not current.valid or current.build == 0:
            continue
        if current.build in snps:
            current.merge(snps_objects=[snps[current.build]])
        snps[current.build] = current
    for build in snps:
        snps[build].sort()
        try:
            save_csv(csv_path, snps[build].snps, user_id, build)
        except Exception as e:
            print(f"Error saving user{user_id}_build{build}_autosomal.csv: {e}")
            continue


def load_txt(txt_path, txt_name):
    snps = SNPs(os.path.join(txt_path, txt_name), resources_dir='../res')
    snps.sort()
    return snps


def get_txt_list(txt_path, user_ids):
    file_list = os.listdir(txt_path)
    txt_list = []
    for file_name in file_list:
        file_extension = file_name.split('.')[-1]
        if file_extension != 'txt':
            continue
        match = re.search(r"user(\d+)_file(\d+)_yearofbirth_(.+)_sex_(.+)\.(.+)\.txt", file_name)
        if match:
            user_id = int(match.group(1))
            if user_id not in user_ids:
                continue
            file_id = int(match.group(2))
            year_of_birth = match.group(3)
            sex = match.group(4)
            provider = match.group(5)
            if provider in ['23andme-exome-vcf']:
                continue
            txt_list.append([user_id, file_id, year_of_birth, sex, provider])
    dataframe = pd.DataFrame(txt_list, columns=['user_id', 'file_id', 'year_of_birth', 'sex', 'provider'])
    dataframe = dataframe.sort_values(by=['user_id', 'file_id'])
    dataframe = dataframe.reset_index(drop=True)
    return dataframe
# endregion


if __name__ == '__main__':
    pd.options.mode.copy_on_write = True
    warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
    csv_dataframe = get_csv_list('../data/opensnp/genotype/csv')
    csv_dataframe = csv_dataframe[csv_dataframe['build_id'] == 37]

