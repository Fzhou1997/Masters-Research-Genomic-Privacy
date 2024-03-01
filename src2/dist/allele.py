import os
import pandas as pd
from tqdm import tqdm

from data.loaders import SNPSLoader


class Distribution:
    def __init__(self, data, build):
        self.data = data
        self.build = build

    def save(self, out: str):
        # make directory if it doesn't exist
        if not os.path.exists(out):
            os.makedirs(out)

        filename = f'alleles_build{self.build}.csv'
        path = os.path.join(out, filename)
        self.data.to_csv(path)


def from_filenames(
    filenames: list[str], hair_colors: list[str], loader: SNPSLoader, verbose=False, rsids=()
) -> dict[int, Distribution]:
    rsids = set(rsids)

    iterator = zip(filenames, hair_colors)
    if verbose:
        iterator = tqdm(iterator)

    # dictionary of build ID -> list of list of genotype alleles
    builds = dict()
    for filename, hair_color in iterator:
        res = loader.load(filename, expand_alleles=True)
        if res is None:
            continue
        snps, build = res

        if rsids:
            snps = snps.loc[list(set(snps.index) & rsids)]
        snps['hair_color'] = hair_color

        if build not in builds:
            builds[build] = []
        builds[build].append(snps)

    # reduce the alleles for each assembly build to counts
    distributions = dict()
    for build in builds.keys():
        data = pd.concat(builds[build])\
            .groupby(['rsid', 'chrom', 'pos', 'hair_color'])[
                ['A', 'C', 'G', 'T', 'D', 'I', '-']]\
            .sum()
        distributions[build] = Distribution(data, build)
    return distributions


def from_csv(out, build) -> Distribution:
    for filename in os.listdir(out):
        if filename.startswith(f'alleles_build{build}'):
            filepath = os.path.join(out, filename)
            data = pd.read_csv(filepath, index_col=0)
            return Distribution(data, build)
    return None
