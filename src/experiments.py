import os
import warnings

import numpy as np
import pandas as pd

from genotype_autosomal import load_txt, load_csv


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


if __name__ == '__main__':
    pd.options.mode.copy_on_write = True
    warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
    snps = load_txt('../data/opensnp/genotype/txt', 'user1_file9_yearofbirth_1985_sex_XY.23andme.txt')
    save_csv('../data/opensnp/genotype/csv', snps.snps, 1, 36)
    snps = load_txt('../data/opensnp/genotype/txt', 'user6_file5_yearofbirth_1959_sex_XY.23andme.txt')
    save_csv('../data/opensnp/genotype/csv', snps.snps, 6, 36)

    df1 = pd.read_csv('../data/opensnp/genotype/csv/user1_build36_autosomal.csv')
    df6 = pd.read_csv('../data/opensnp/genotype/csv/user6_build36_autosomal.csv')

    print(df1.shape, df6.shape)

    concatenated = pd.concat([df1, df6])
    summed = concatenated.groupby('rsid').sum()

    print(concatenated.groupby('rsid').sum().head(15))
