import os

from pandas import DataFrame
from tqdm import tqdm

from legacy.src2.data.loaders import SNPSLoader


class Distribution:
    def __init__(self, data: DataFrame, build: int):
        self.data = data
        self.build = build

    def save(self, out: str):
        # make directory if it doesn't exist
        if not os.path.exists(out):
            os.makedirs(out)

        filename = f'genotype_distribution_build{self.build}.csv'
        path = os.path.join(out, filename)
        self.data.to_csv(path)


# expects preprocessed genotype files in the following format:
# user#_build#_autosomal.csv
# rsid  chrom   pos     alleles
# {int} {int}   {int}   {str}
def from_users(
        build: int,
        users: list[int],
        hair_colors: list[str],
        loader: SNPSLoader,
        verbose=False,
        rsids=()) -> dict[int, Distribution]:
    rsids = set(rsids)

    iterator = zip(filenames, hair_colors)
    if verbose:
        iterator = tqdm(iterator)


