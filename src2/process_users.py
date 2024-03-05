import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import dist
from data import loaders


def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        help="Assembly build to use, usually can be 36/37/38",
        default=37, type=lambda value: int(value))
    parser.add_argument(
        "--raw",
        help="Directory where raw genotype files are stored",
        default='data/opensnp/raw')
    parser.add_argument(
        "--out",
        help="Directory where output allele files are stored",
        default='data/opensnp/out')
    parser.add_argument(
        "--res",
        help="Directory where resource files are stored",
        default='data/res')
    return parser


if __name__ == '__main__':
    parser = make_arg_parser()
    args = parser.parse_args()

    # get the unique user files from our phenotype loader
    # - this step ensures we only look at files with phenotype labels
    phenotype_loader = loaders.PhenotypeLoader(
        args.raw, id=202402130100, verbose=True)
    filenames = [
        phenotype_loader.get_filename(user_id)
        for user_id in phenotype_loader.df.index
    ]
    hair_colors = phenotype_loader.df.hair_color

    # load in common RSIDs to trim genotype files with
    with open(os.path.join(args.out, f'common_rsids_build{args.build}.txt'), 'r') as f:
        rsids = {rsid.strip() for rsid in f.readlines()}

    snps_loader = loaders.SNPSLoader(args.raw, res_path=args.res)

    # load in our allele distribution
    distribution = dist.allele.from_csv(args.out, args.build)
    if not distribution:
        distributions = dist.allele.from_filenames(
            filenames, phenotype_loader.df.hair_color, snps_loader, verbose=True, rsids=rsids)
        for build in distributions:
            distributions[build].save(args.out)
        distribution = distributions[build]

    # find the alleles that occur the most for each RSID
    reference_alleles = distribution.data\
        .groupby('rsid')[['A', 'C', 'G', 'T', 'D', 'I']]\
        .sum()\
        .idxmax(axis=1)

    # collect formatted user vectors
    matrix_filename = os.path.join(args.out, f'matrix_build{args.build}.csv')
    try:
        matrix = pd.read_csv(matrix_filename, index_col=0)
    except:
        user_ids = []
        user_vectors = []
        for user_id in tqdm(phenotype_loader.df.index):
            filename = phenotype_loader.get_filename(user_id)
            res = snps_loader.load(filename, expand_alleles=True)
            if res is None:
                continue
            snps, build = res
            if build != args.build:
                continue
            # trim down by common RSIDs!
            snps = snps.loc[list(set(snps.index) & rsids)]

            user_vector = []
            for rsid, allele in zip(reference_alleles.index, reference_alleles.values):
                # -1 - missing data
                #  0 - all reference
                #  1 - one alternate
                #  2 - two alternates
                if rsid not in snps.index or snps.loc[rsid, '-'] > 0:
                    user_vector.append(np.nan)
                else:
                    user_vector.append(2 - snps.loc[rsid, allele])
            user_vector = pd.Series(
                user_vector, index=reference_alleles.index, name=user_id)
            user_ids.append(user_id)
            user_vectors.append(user_vector)

        matrix = pd.concat(user_vectors, axis=1).T
        matrix.to_csv(os.path.join(args.out, f'matrix_build{args.build}.csv'))

        # make our probability matrices
        colors, counts = np.unique(hair_colors, return_counts=True)

"""
    we want, for each RSID...
        p(0|black), p(0|brown), p(0|blonde)
        p(1|black), p(1|brown), p(1|blonde)
        p(2|black), p(2|brown), p(2|blonde)
    and we know
        p(n|h) = #(n,h)/#(h)
"""
