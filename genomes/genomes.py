import os

import pandas as pd
from torch import tensor


class Genomes:
    def __init__(self, build):
        self.build = build
        self.num_genotypes = 0
        self.genotypes = pd.DataFrame()
        self.phenotypes = pd.DataFrame()

    def concat_genotype(self, genotype):
        if (genotype.get_build() != self.build):
            raise ValueError('Reference genome build mismatch')
        self.num_genotypes += 1
        user_id = genotype.get_user_id()
        user_genotype = genotype.get_genotype().T
        user_genotype.index = pd.Index([user_id], name='user_id')
        self.genotypes = pd.concat([self.genotypes, user_genotype], ignore_index=False)

    def concat_phenotypes(self, phenotypes):
        self.phenotypes = pd.concat([self.phenotypes, phenotypes.get_phenotypes()], ignore_index=False)

    def filter_genotypes_na(self):
        self.genotypes = self.genotypes.dropna(axis='columns')

    def filter_phenotypes_na(self):
        self.phenotypes = self.phenotypes.dropna(axis='rows')

    def filter_phenotypes_genotypes(self):
        self.phenotypes = self.phenotypes.loc[self.genotypes.index]

    def get_build(self):
        return self.build

    def get_num_genotypes(self):
        return self.num_genotypes

    def get_user_ids(self):
        return self.genotypes.index

    def get_rsids(self):
        return self.genotypes.columns

    def get_genotypes(self):
        return self.genotypes

    def get_phenotypes(self):
        return self.phenotypes

    def get_genomes(self):
        return pd.concat([self.genotypes, self.phenotypes], axis=1, ignore_index=False)

    def get_genotypes_tensor(self):
        genotypes_values = self.genotypes.values
        return tensor(genotypes_values)

    def get_phenotypes_tensor(self):
        phenotypes_values = self.phenotypes.values
        return tensor(phenotypes_values)

    def get_tensor(self):
        genomes_values = self.get_genomes().values
        return tensor(genomes_values)

    def sort_rsids(self, rsids):
        self.genotypes = self.genotypes[rsids]

    def save(self, out_path):
        os.makedirs(out_path, exist_ok=True)
        genotypes_out = self.genotypes.reset_index()
        phenotypes_out = self.phenotypes.reset_index()
        genotypes_out.to_csv(os.path.join(out_path, f'build_{self.build}_count_{self.num_genotypes}_genomes_genotype.csv'), index=False)
        phenotypes_out.to_csv(os.path.join(out_path, f'build_{self.build}_count_{self.num_genotypes}_genomes_phenotype.csv'), index=False)

    def load(self, data_path):
        genotype_loaded = False
        phenotype_loaded = False
        for file_name in os.listdir(data_path):
            if file_name.startswith(f'build_{self.build}_') and file_name.endswith('_genomes_genotype.csv'):
                self.num_genotypes = int(file_name.split('_')[3])
                self.genotypes = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
                genotype_loaded = True
            if file_name.startswith(f'build_{self.build}_') and file_name.endswith('_genomes_phenotype.csv'):
                self.num_genotypes = int(file_name.split('_')[3])
                self.phenotypes = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
                phenotype_loaded = True
            if genotype_loaded and phenotype_loaded:
                if len(self.genotypes) != len(self.phenotypes):
                    raise ValueError('Genotype and phenotype mismatch')
                else:
                    return
        raise FileNotFoundError('No genomes files found')

