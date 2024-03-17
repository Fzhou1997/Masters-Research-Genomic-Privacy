import itertools
import os

import pandas as pd

ALLELES = 'ACGTDI-'
GENOTYPES = [''.join(item) for item in itertools.product('ACGTDI', repeat=2)] + ['--']
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

    def save(self, out_path):
        out = self.genotypes.reset_index()
        os.makedirs(out_path, exist_ok=True)
        out.to_csv(os.path.join(out_path, f'build_{self.build}_count_{self.num_genotypes}_genotypes.csv'), index=False)

    def load(self, data_path):
        for file_name in os.listdir(data_path):
            if file_name.startswith('build_') and file_name.endswith('_genotypes.csv'):
                self.build = int(file_name.split('_')[1])
                self.num_genotypes = int(file_name.split('_')[3])
                self.genotypes = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
                self.probabilities = self.genotypes.div(self.genotypes.sum(axis=1), axis=0)
                return