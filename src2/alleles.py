import argparse
import os
import pandas as pd

import dist
from proc import loaders


def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        help="Assembly build to use, usually can be 36/37/38",
        default=37, type=lambda value: int(value))
    parser.add_argument(
        "--threshold",
        help="Threshold to filter how often RSIDs must occur amongst genotypes",
        default=0.97, type=lambda value: float(value))
    parser.add_argument(
        "--raw",
        help="Directory where raw genotype files are stored",
        default='data/opensnp/raw')
    parser.add_argument(
        "--out",
        help="Directory where output RSID files are stored",
        default='data/opensnp/out')
    parser.add_argument(
        "--res",
        help="Directory where resource files are stored",
        default='data/res')
    return parser


if __name__ == '__main__':
    parser = make_arg_parser()
    args = parser.parse_args()

    # load in our allele distribution
    distribution = dist.allele.from_csv(args.out, args.build)
    if not distribution:
        # get the unique user IDs from our phenotype loader
        # - this step ensures we only look at files with phenotype labels
        phenotype_loader = loaders.PhenotypeLoader(
            args.raw, id=202402130100, verbose=True)
        filenames = [
            phenotype_loader.get_filename(user_id)
            for user_id in phenotype_loader.df.index
        ]

        rsids = ()
        with open(os.path.join(args.out, f'common_rsids_build{args.build}.txt'), 'r') as f:
            rsids = [rsid.strip() for rsid in f.readlines()]

        loader = loaders.SNPSLoader(args.raw, res_path=args.res)
        distributions = dist.allele.from_filenames(
            filenames, loader, verbose=True, rsids=rsids)

        for build in distributions:
            distributions[build].save(args.out)

"""
we want to impute missing alleles into RSID locations for a particular user
we want, for any particular RSID, to pick allele with highest:

we want P(allele|hair)

we have P(allele)
we have P(hair)
we can calculate P(hair|allele)

then because we know because:
P(allele|hair)P(hair) = P(hair|allele)P(allele)
it follows that
P(allele|hair) = P(hair|allele)P(allele) / P(hair)
"""
