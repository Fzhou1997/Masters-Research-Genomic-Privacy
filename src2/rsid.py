"""rsids.py

This script is used to extract information about the RSIDs in our OpenSNP
dataset. We extract the distribution of RSIDs throughout the genotype files,
and save the most common occuring ones for us to use further downstream in our
machine learning tasks.

This script should be executed from the root directory, and you can see the
optional arguments with the --help flag.
"""
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

    # get the unique user IDs from our phenotype loader
    # - this step ensures we only look at files with phenotype labels
    phenotype_loader = loaders.PhenotypeLoader(
        args.raw, id=202402130100, verbose=True)
    filenames = [
        phenotype_loader.get_filename(user_id)
        for user_id in phenotype_loader.df.index
    ]

    distribution = dist.rsid.from_csv(args.out, build=args.build)

    if not distribution:
        # load the files, this one might take some time if they're not saved already (~20mins)
        print('Distribution not found, parsing raw genotypes...')
        snps_loader = loaders.SNPSLoader(args.raw, res_path=args.res)
        distributions = dist.rsid.from_filenames(
            filenames, snps_loader, verbose=True)

        for build in distributions.keys():
            distributions[build].save(args.out)

        distribution = distributions[args.build]

    print('Distribution(s) loaded.')

    # get the RSIDs that are shared amongst 97% of the genotype samples
    normalized = distribution.data.astype(float) / distribution.count
    filtered = normalized[normalized > args.threshold]
    common_rsids = sorted(list(filtered.index))

    # save common RSIDs to output directory
    path = os.path.join(args.out, f'common_rsids_build{args.build}.txt')
    with open(path, 'w') as f:
        for rsid in common_rsids:
            f.write(f'{rsid}\n')

    print('Most common RSIDs saved. Exiting.')
