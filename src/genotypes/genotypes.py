import itertools
import os

import numpy as np
import pandas as pd


class Genotypes:
    def __init__(self):
        self.genotype_counts = None

    def append(self, genotype_counts):
        if not self.genotype_counts:
            self.genotype_counts = genotype_counts
        else:
            self.genotype_counts = pd.concat([self.genotype_counts, genotype_counts])

    def get_genotype_counts(self):
        self.genotype_counts = self.genotype_counts.groupby(level='rsid').sum()
        return self.genotype_counts.copy()

    def get_genotype_probabilities(self):
        genotype_probabilities = self.get_genotype_counts()
        genotype_probabilities = genotype_probabilities.div(genotype_probabilities.sum(axis=1), axis=0)
        return genotype_probabilities

    def get_reference_alleles(self):
        genotype_counts = self.get_genotype_counts()
        genotypes = list(itertools.product('ACGTDI', repeat=2))
        genotypes.append('--')
        alleles = list('ACGTDI-')
        transformation_matrix = np.zeros((len(genotypes), len(alleles)))
        for i, genotype in enumerate(genotypes):
            for j, allele in enumerate(alleles):
                transformation_matrix[i, j] = genotype.count(allele)
        allele_counts = genotype_counts.dot(transformation_matrix)
        reference_alleles = allele_counts.idxmax(axis=1)
        return reference_alleles

    def save(self, out_path):
        self.genotype_counts = self.genotype_counts.groupby(level='rsid').sum()
        out = self.genotype_counts.reset_index()
        out.to_csv(os.path.join(out_path, f'genotype_counts.csv'), index=False)

    def load(self, data_path):
        self.genotype_counts = pd.read_csv(os.path.join(data_path, 'genotype_counts.csv'), index_col=0)