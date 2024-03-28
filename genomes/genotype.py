import itertools
import logging
import os
import warnings

import pandas as pd
from snps import SNPs

RSIDS_INVALID = r'[^0-9a-z]+'
CHROMOSOMES_AUTOSOMAL = set(str(i) for i in range(1, 23))
CHROMOSOMES_INVALID = r'[^0-9XYMT]+'
ALLELES = 'ACGTDI-'
ALLELES_INVALID = r'[^ACGTDI\-0]+'
ALLELES_NA = r'.*[-0]+.*'
GENOTYPES = [''.join(item) for item in itertools.product('ACGT', repeat=2)] + [''.join(item) for item in itertools.product('DI', repeat=2)] + ['--']



class Genotype:
    def __init__(self):
        self.user_id = None
        self.build = None
        self.genotype = None

    def from_user_id(self, data_path, res_path, user_id, build):
        user_file_names = [file_name for file_name in os.listdir(data_path)
                           if file_name.startswith(f'user{user_id}_')
                           and file_name.endswith('.txt')
                           and 'exome-vcf' not in file_name]
        if not user_file_names:
            raise FileNotFoundError(f'No genotype files found')
        snps = None
        for file_name in user_file_names:
            file_path = os.path.join(data_path, file_name)
            try:
                s = SNPs(file_path, resources_dir=res_path)
            except Exception as e:
                continue
            if not s.valid or not s.build_detected or s.build != build:
                continue
            if snps is not None:
                s.merge(snps_objects=[snps])
            snps = s
        if snps is None:
            raise ValueError('No valid genotype files found')
        snps.sort()
        self.user_id = user_id
        self.build = build
        self.genotype = snps.snps

    def clean(self):
        self.genotype['genotype'] = self.genotype['genotype'].str.upper()
        self.genotype['genotype'] = self.genotype['genotype'].str.replace(ALLELES_INVALID, '', regex=True)
        self.genotype['genotype'] = self.genotype['genotype'].str.replace(ALLELES_NA, '--', regex=True)
        self.genotype['genotype'] = self.genotype['genotype'].fillna('--')
        self.genotype['chrom'] = self.genotype['chrom'].str.upper()
        self.genotype['chrom'] = self.genotype['chrom'].str.replace(CHROMOSOMES_INVALID, '', regex=True)
        self.genotype.index = self.genotype.index.str.lower()
        self.genotype.index = self.genotype.index.str.replace(RSIDS_INVALID, '', regex=True)
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def filter_chromosomes_autosomal(self):
        self.genotype = self.genotype[self.genotype['chrom'].isin(CHROMOSOMES_AUTOSOMAL)]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def filter_chromosomes(self, chromosomes):
        self.genotype = self.genotype[self.genotype['chrom'].isin(chromosomes)]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def filter_rsids_proprietary(self):
        self.genotype = self.genotype.loc[self.genotype.index.str.startswith('rs')]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def filter_rsids(self, rsids):
        self.genotype = self.genotype.loc[self.genotype.index.isin(rsids)]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def is_valid(self):
        return len(self.genotype.index) > 0

    def is_imputed(self):
        return '--' not in self.genotype['genotype'].values

    def is_alternate_allele_count_encoded(self):
        return self.genotype['genotype'].isin(range(3)).all()

    def get_user_id(self):
        return self.user_id

    def get_build(self):
        return self.build

    def get_rsids(self):
        return self.genotype[['chrom', 'pos']]

    def get_genotype(self):
        return self.genotype[['genotype']]

    def get_one_hot(self):
        one_hot = self.genotype.loc[:, []]
        for genotype in GENOTYPES:
            one_hot[genotype] = (self.genotype['genotype'] == genotype).astype(int)
        return one_hot

    def get_allele_counts(self):
        allele_counts = self.genotype.loc[:, []]
        for allele in ALLELES:
            allele_counts[allele] = self.genotype['genotype'].apply(lambda genotype: genotype.count(allele))
        return allele_counts

    def impute_bayesian(self, genotypes):
        target_rsids = self.genotype[self.genotype['genotype'] == '--'].index
        genotype_counts = genotypes.get_genotype_counts()
        genotype_counts = genotype_counts.drop(columns=['--'])
        genotype_modes = genotype_counts.idxmax(axis=1)
        for rsid in target_rsids:
            self.genotype.at[rsid, 'genotype'] = genotype_modes[rsid]

    def encode_alternate_allele_count(self, reference_alleles):
        for rsid in self.genotype.index:
            reference_allele = reference_alleles[rsid]
            if self.genotype.at[rsid, 'genotype'] == '--':
                self.genotype.at[rsid, 'genotype'] = -1
            else:
                self.genotype.at[rsid, 'genotype'] = self.genotype.at[rsid, 'genotype'].count(reference_allele)

    def save(self, out_path):
        out = self.genotype.reset_index()
        os.makedirs(out_path, exist_ok=True)
        out.to_csv(os.path.join(out_path, f'user{self.user_id}_build{self.build}.csv'), index=False)

    def load(self, data_path, user_id, build):
        file_path = os.path.join(data_path, f'user{user_id}_build{build}.csv')
        self.user_id = user_id
        self.build = build
        self.genotype = pd.read_csv(file_path, index_col=0)
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def remove(self, data_path):
        file_path = os.path.join(data_path, f'user{self.user_id}_build{self.build}.csv')
        if os.path.exists(file_path):
            os.remove(file_path)
