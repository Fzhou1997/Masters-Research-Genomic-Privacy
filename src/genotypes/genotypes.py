import itertools
import os

import pandas as pd

ALLELES = 'ACGTDI-'
GENOTYPES = [''.join(item) for item in itertools.product('ACGT', repeat=2)] + [''.join(item) for item in itertools.product('DI', repeat=2)] + ['--']
TRANSFORMATION = pd.DataFrame(columns=list(ALLELES), index=GENOTYPES)
for genotype in GENOTYPES:
    for allele in ALLELES:
        TRANSFORMATION.at[genotype, allele] = genotype.count(allele)


class Genotypes:
    def __init__(self, build):
        self.build = build
        self.num_genotypes = 0
        self.genotypes = pd.DataFrame()

    def concat_genotype(self, genotype):
        if genotype.get_build() != self.build:
            raise ValueError('Reference genome build mismatch')
        self.num_genotypes += 1
        one_hot = genotype.get_one_hot()
        self.genotypes = pd.concat([self.genotypes, one_hot], ignore_index=False)
        self.genotypes = self.genotypes.groupby(level='rsid').sum()

    def concat_genotypes(self, genotypes):
        for genotype in genotypes:
            if genotype.get_build() != self.build:
                raise ValueError('Reference genome build mismatch')
            self.num_genotypes += 1
            one_hot = genotype.get_one_hot()
            self.genotypes = pd.concat([self.genotypes, one_hot], ignore_index=False)
        self.genotypes = self.genotypes.groupby(level='rsid').sum()

    def get_build(self):
        return self.build

    def get_num_genotypes(self):
        return self.num_genotypes

    def get_genotype_counts(self):
        return self.genotypes

    def get_genotype_probabilities(self):
        return self.genotypes.div(self.genotypes.sum(axis=1), axis=0)

    def get_allele_counts(self):
        return self.genotypes.dot(TRANSFORMATION)

    def get_reference_alleles(self):
        return self.get_allele_counts().idxmax(axis=1)

    def save(self, out_path, file_name=None):
        out = self.genotypes.reset_index()
        os.makedirs(out_path, exist_ok=True)
        if file_name is None:
            file_name = f'build_{self.build}_count_{self.num_genotypes}_genotypes.csv'
        out.to_csv(os.path.join(out_path, file_name), index=False)

    def load(self, data_path, file_name=None):
        if file_name is None:
            for file_name in os.listdir(data_path):
                if file_name.startswith(f'build_{self.build}') and file_name.endswith('_genotypes.csv'):
                    self.num_genotypes = int(file_name.split('_')[3])
                    self.genotypes = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
                    return
            raise FileNotFoundError('No genotypes file found')
        else:
            if os.path.exists(os.path.join(data_path, file_name)):
                self.genotypes = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
            else:
                raise FileNotFoundError('No genotypes file found')
