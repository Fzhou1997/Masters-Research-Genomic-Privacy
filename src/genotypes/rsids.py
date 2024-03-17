import os.path

import pandas as pd


class Rsids:
    def __init__(self, build):
        self.build = build
        self.num_genotypes = 0
        self.rsids = pd.DataFrame()

    def concat_genotype(self, genotype):
        if genotype.get_build() != self.build:
            raise ValueError('Reference genome build mismatch')
        self.num_genotypes += 1
        rsids = genotype.get_rsids()
        rsids['count'] = 1
        self.rsids = pd.concat([self.rsids, rsids], ignore_index=False)
        self.rsids = self.rsids.groupby(['rsid', 'chrom', 'pos']).sum().reset_index()
        self.rsids = self.rsids.set_index('rsid')

    def get_build(self):
        return self.build

    def get_num_genotypes(self):
        return self.num_genotypes

    def get_rsid_counts(self):
        return self.rsids[['count']]

    def get_rsid_map(self):
        return self.rsids[['chrom', 'pos']]

    def get_rsid_probabilities(self):
        return self.rsids[['count']] / self.num_genotypes

    def get_common_rsids(self):
        return self.rsids[self.rsids['count'] == self.num_genotypes].index

    def save(self, out_path):
        out = self.rsids.reset_index()
        os.makedirs(out_path, exist_ok=True)
        out.to_csv(os.path.join(out_path, f'build_{self.build}_count_{self.num_genotypes}_rsids.csv'), index=False)

    def load(self, data_path):
        for file_name in os.listdir(data_path):
            if file_name.startswith('build_') and file_name.endswith('_rsids.csv'):
                self.build = int(file_name.split('_')[1])
                self.num_genotypes = int(file_name.split('_')[3])
                self.rsids = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
                return
