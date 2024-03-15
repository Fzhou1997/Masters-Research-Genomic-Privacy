import os.path

import numpy as np
import pandas as pd


class Rsids:
    def __init__(self, build):
        self.build = build
        self.num_genotypes = 0
        self.rsid_map = pd.DataFrame()
        self.rsid_counts = pd.DataFrame(columns=['count'])

    def concat_genotype(self, genotype, rsids):
        if genotype.get_build() != self.build:
            raise ValueError('Reference genome build mismatch')
        self.num_genotypes += 1

        rsids = rsids.index
        rsids = rsids.to_frame()
        self.rsid_counts = self.rsid_counts.add(rsids, fill_value=0)

    def get_rsid_counts(self):
        return self.rsid_counts.copy()

    def get_rsid_probabilities(self):
        rsids_probabilities = self.get_rsid_counts()
        rsids_probabilities.rename(columns={'count': 'probability'}, inplace=True)
        rsids_probabilities = rsids_probabilities.div(self.num_genotypes)
        return rsids_probabilities

    def get_common_rsids(self):
        rsid_counts = self.get_rsid_counts()
        return rsid_counts[rsid_counts['count'] == self.num_genotypes].index

    def save(self, out_path):
        out = self.rsid_counts.reset_index()
        out.to_csv(os.path.join(out_path, f'user_count_{self.num_genotypes}_rsid_counts.csv'), index=False)

    def load(self, data_path):
        for file_name in os.listdir(data_path):
            if file_name.startswith('user_count_') and file_name.endswith('_rsid_counts.csv'):
                self.rsid_counts = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
                self.num_genotypes = int(file_name.split('_')[2])
                return
