import os

import pandas as pd


class Genomes:
    def __init__(self, build):
        self.build = build
        self.num_genotypes = 0
        self.num_phenotypes = 0
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
        self.num_phenotypes += 1
        self.phenotypes = pd.concat([self.phenotypes, phenotypes], ignore_index=False)

    def filter_genotypes_na(self):
        self.genotypes = self.genotypes.dropna(axis='columns')

    def filter_phenotypes_na(self):
        self.phenotypes = self.phenotypes.dropna(axis='rows')

    def get_build(self):
        return self.build

    def get_num_genotypes(self):
        return self.num_genotypes

    def get_num_phenotypes(self):
        return self.num_phenotypes

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

    def save(self, out_path):
        out = self.get_genomes().reset_index()
        os.makedirs(out_path, exist_ok=True)
        out.to_csv(os.path.join(out_path, f'build_{self.build}_genotype_count_{self.num_genotypes}_phenotype_count_{self.num_phenotypes}_genomes.csv'), index=False)

    def load(self, data_path):
        for file_name in os.listdir(data_path):
            if file_name.startswith('build_') and file_name.endswith('_genomes.csv'):
                self.build = int(file_name.split('_')[1])
                self.num_genotypes = int(file_name.split('_')[3])
                self.num_phenotypes = int(file_name.split('_')[5])
                genomes = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
                self.genotypes = genomes[genomes.columns[:-self.num_phenotypes]]
                self.genotypes = self.genotypes.dropna(axis='rows')
                self.phenotypes = genomes[genomes.columns[-self.num_phenotypes:]]
                return
