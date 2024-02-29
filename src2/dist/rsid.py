import os
import re
import numpy as np
import pandas as pd
from pandas import Series
from collections import Counter
from tqdm import tqdm

from proc.loaders import SNPSLoader


class Distribution:
    def __init__(self, data: Series, count: int, build: int):
        self.data = data
        self.count = count
        self.build = build

    def save(self, out: str):
        # make directory if it doesn't exist
        if not os.path.exists(out):
            os.makedirs(out)

        filename = f'rsids_build{self.build}_{self.count}samples.csv'
        path = os.path.join(out, filename)
        self.data.to_csv(path, header=None)


def from_filenames(
    filenames: list[str], loader: SNPSLoader, verbose=False
) -> dict[int, Distribution]:
    """Generate a dictionary of RSID distributions from a list of filenames.

    Args:
        filenames (list[str]): the names of the genotype files.
        loader (SNPSLoader): the SNP loader object.
        verbose (bool, optional): whether to print debug statements.

    Returns:
        dict[int, Distribution]: a dictionary mapping build IDs to distributions.
    """

    iterator = iter(filenames)
    if verbose:
        iterator = tqdm(iterator)
    # dictionary of build ID -> list of list of genotype RSIDs
    builds = dict()
    for filename in iterator:
        res = loader.load(filename)
        if res is None:
            continue
        snps, build = res
        if build not in builds:
            builds[build] = []
        builds[build].append(np.array(snps.index))

    # reduce the RSIDs for each assembly build to its number
    # of occurences across samples
    reduced_builds = dict()
    build_counts = dict()  # keep track of how many samples successfully read per build
    for build in builds.keys():
        # stack all RSIDs horizontally, and wrap them in a Counter object
        # to count the occurrences
        reduced_builds[build] = Counter(np.hstack(builds[build]))

        # the number of lists for each build is the number of samples parsed
        build_counts[build] = len(builds[build])

    # convert counter dictionary to pandas dataframe
    distributions = dict()
    for build in reduced_builds.keys():
        data = Series(reduced_builds[build])
        count = build_counts[build]
        distributions[build] = Distribution(data, count, build)

    return distributions


def from_csv(out, build) -> Distribution:
    """Load an RSID distribution from a generated CSV file.

    Args:
        filepath (str): the full path to the CSV file.

    Returns:
        Distribution: the RSID distribution object.
    """
    for filename in os.listdir(out):
        if filename.startswith(f'rsids_build{build}'):
            result = re.match(r'rsids_build\d+_(\d+)samples.csv', filename)
            count = int(result.group(1))
            filepath = os.path.join(out, filename)
            data = pd.read_csv(filepath, index_col=0, header=None,
                               names=[result.group(0)]).squeeze()
            return Distribution(data, count, build)
    return None
