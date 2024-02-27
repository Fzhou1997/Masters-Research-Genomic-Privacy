import os
import numpy as np
import pandas as pd

from genotype_autosomal import load_csv, get_csv_list
import multiprocessing as mp


class Vectorizer:
    def __init__(self, csv_path, npy_path, build_id, reference_allele_indices, reference_genome):
        self.csv_path = csv_path
        self.npy_path = npy_path
        self.build_id = build_id
        self.reference_allele_indices = reference_allele_indices
        self.reference_genome = reference_genome

    def __call__(self, user_id):
        genome_df = load_csv(self.csv_path, f"user{user_id}_build{self.build_id}_autosomal.csv")
        genome_df = genome_df.loc[self.reference_genome.index]
        difference_df = self.reference_genome - genome_df
        genome_list = []
        for index, reference_allele_index in self.reference_allele_indices.items():
            if genome_df.loc[index, '-'] > 0:
                genome_list.append(-1)
            else:
                genome_list.append(difference_df.loc[index, reference_allele_index])
        genome_array = np.array(genome_list).astype(int)
        np.save(os.path.join(self.npy_path, f"user{user_id}_build{self.build_id}_autosomal.npy"), genome_array)


def process_users_npys(csv_path, npy_path, user_ids, build_id, reference_allele_indices, reference_genome, n_proc=1):
    if n_proc > 1:
        with mp.Pool(n_proc) as p:
            p.map(Vectorizer(csv_path, npy_path, build_id, reference_allele_indices, reference_genome), user_ids)
    else:
        vec = Vectorizer(csv_path, npy_path, build_id, reference_allele_indices, reference_genome)
        for user_id in user_ids:
            vec(user_id)


if __name__ == "__main__":
    # read in the CSVs of our allele counts
    csv_summary = pd.read_csv('../data/opensnp/genotype/sum/csv_summary.csv', index_col=0)
    print(csv_summary.head(10))
    print('\n-------\n')

    # calculate allele frequencies
    allele_frequencies = csv_summary.div(csv_summary.sum(axis=1), axis=0)
    allele_frequencies.to_csv('../data/opensnp/genotype/sum/allele_frequencies.csv')
    print(allele_frequencies.head(10))
    print('\n-------\n')

    # save reference alleles to a CSV
    reference_alleles = csv_summary.idxmax(axis=1)
    reference_alleles.to_csv('../data/opensnp/genotype/sum/reference_allele_indices.csv', header=False)
    print(reference_alleles.head(10))
    print('\n-------\n')

    # save reference genome to CSV
    reference_genome_df = pd.DataFrame(index=csv_summary.index, columns=csv_summary.columns)
    reference_genome_df = reference_genome_df.fillna(0)
    for index, reference_allele_index in reference_alleles.items():
        reference_genome_df.loc[index, reference_allele_index] = 2
    reference_genome_df.to_csv('../data/opensnp/genotype/sum/reference_genome.csv')
    print(reference_genome_df.head(10))
    print('\n-------\n')

    csv_list = get_csv_list('../data/opensnp/genotype/csv')
    csv_list = csv_list[csv_list['build_id'] == 37]
    user_ids = csv_list['user_id'].unique()
    process_users_npys('../data/opensnp/genotype/csv', '../data/opensnp/genotype/npy', user_ids, 37, reference_alleles,
                       reference_genome_df, n_proc=os.cpu_count() - 1)

    # reference_genome
    #      a c g t i d -
    # rsid0 0 2 0 0 0 0 0

    # -

    # genotype
    #      a c g t i d -
    # rsid0 0 1 0 1 0 0 0

    # =

    # intermediate
    #      a c g t i d -
    # rsid0 0 1 0-1 0 0 0

    # npy = (reference_genome - genotype)[reference_alleg_indices].to_numpy()
