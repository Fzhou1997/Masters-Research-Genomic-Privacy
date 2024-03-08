import itertools
import os

import pandas as pd
from numpy import random
from pandas import DataFrame, Series
from snps import SNPs


class GenotypeProcessor:
    def __init__(self, raw_path: str, res_path: str, out_path: str, build: int):
        self.raw_path = raw_path
        self.res_path = res_path
        self.out_path = out_path
        self.raw_filenames = os.listdir(self.raw_path)
        self.processed_filenames = os.listdir(self.out_path)
        self.build = build
        self.chromosomes = set(range(1, 23))
        # self.genotype_distribution = None
        # self.allele_distribution = None
        self.alleles = ['A', 'C', 'G', 'T', 'D', 'I']
        self.genotype_combinations = list(
            itertools.product(self.alleles, repeat=2))
        self.genotype_combinations.append('--')

    # def set_genotype_distribution(self, genotype_distribution: GenotypeDistribution):
    #     self.genotype_distribution = genotype_distribution

    # def set_allele_distribution(self, allele_distribution: AlleleDistribution):
    #     self.allele_distribution = allele_distribution

    def load_raw(self, user_id: int) -> DataFrame:
        """Loads the raw genotype data for a given user."""
        user_filenames = [filename for filename in self.raw_filenames
                          if filename.startswith(f'user{user_id}')
                          and filename.endswith('.txt')
                          and 'exome-vcf' not in filename]
        if not user_filenames:
            raise FileNotFoundError(
                f'No genotype files found for user {user_id}')
        snps = None
        for filename in user_filenames:
            path = os.path.join(self.raw_path, filename)
            try:
                s = SNPs(path, resources_dir=self.res_path)
            except Exception as e:
                print(f'Error loading {filename}: {e}')
                continue
            if not s.valid or not s.build_detected or s.build != self.build:
                continue
            if snps is None:
                snps = s
            else:
                snps.merge(snps_objects=[s])
        if snps is None:
            raise ValueError(
                f'No valid genotype files found for user {user_id}')
        snps.sort()
        genotype_df = snps.snps
        genotype_df = genotype_df[genotype_df['chrom'].isin(self.chromosomes)]
        genotype_df['genotype'] = genotype_df['genotype'].apply(
            lambda x: '--' if '-' in x else x)
        for combination in self.genotype_combinations:
            genotype_df[combination] = (
                genotype_df['genotype'] == combination).astype(int)
        return genotype_df

    def load_processed(self, user_id: int) -> DataFrame:
        """Loads the processed genotype data for a given user."""
        user_filename = f'user{user_id}_build{self.build}_autosomal.csv'
        if user_filename not in self.processed_filenames:
            raise FileNotFoundError(
                f'No processed genotype file found for user {user_id}')
        user_filepath = os.path.join(self.out_path, user_filename)
        genotype_df = pd.read_csv(user_filepath)
        return genotype_df

    def save_processed(self, user_id: int, genotype_df: DataFrame):
        """Saves the processed genotype data for a given user."""
        user_filename = f'user{user_id}_build{self.build}_autosomal.csv'
        user_filepath = os.path.join(self.out_path, user_filename)
        genotype_df.to_csv(user_filepath, index=False)

    def impute(self, genotype_df: DataFrame, hair_color: str) -> DataFrame:
        """Imputes missing genotype and appends the imputed genotype to the dataframe."""
        genotype_df['genotype_imputed'] = genotype_df['genotype']
        for rsid, row in genotype_df.iterrows():
            if row['genotype'] == '--':
                genotype_imputed = random.choice(
                    a=self.genotype_combinations, p=self.genotype_distribution[hair_color][rsid])
                genotype_df.at[rsid, 'genotype_imputed'] = genotype_imputed
        return genotype_df

    def expand(self, genotype: DataFrame) -> DataFrame:
        """Appends individual alleles and allele one hot encoding to the dataframe."""
        expanded = genotype.copy()  # should this be deep or shallow copy?
        expanded['allele1'] = genotype.apply(
            lambda row: row['genotype'][0], axis=1)
        expanded['allele2'] = genotype.apply(
            lambda row: row['genotype'][1], axis=1)
        for allele in self.alleles:
            expanded[allele] = genotype['genotype'].count(allele)
        return expanded

    def encode(self, genotype: DataFrame, reference_alleles: Series) -> DataFrame:
        """Appends alternate allele counts to the dataframe"""
        encoded = genotype.copy()  # should this be deep or shallow copy?
        for index, allele in reference_alleles.items():
            encoded.loc[index, 'alternate_allele_count'] =\
                2 - genotype.loc[index, 'genotype'].count(allele)
        return encoded
