import os.path
from collections import Counter
from os import PathLike

import pandas as pd

from genomes import Genotype


class Rsids:
    def __init__(self, build: int):
        self.build = build
        self.num_genotypes = 0
        self.rsid_counts = Counter()
        self.rsid_map = {}

    def concat_genotype(self, genotype: Genotype) -> None:
        if genotype.get_build() != self.build:
            raise ValueError('Reference genome build mismatch')
        self.num_genotypes += 1
        rsids = genotype.get_rsids()
        self.rsid_counts.update(rsids.index)
        self.rsid_map.update(rsids.to_dict(orient='index'))

    def concat_genotypes(self, genotypes: list[Genotype]) -> None:
        for genotype in genotypes:
            self.concat_genotype(genotype)

    def get_build(self) -> int:
        return self.build

    def get_num_genotypes(self) -> int:
        return self.num_genotypes

    def get_rsid_counts(self) -> Counter[str]:
        return self.rsid_counts

    def get_rsid_map(self) -> dict[str, dict[str, int]]:
        return self.rsid_map

    def get_rsid_probabilities(self):
        return {
            rsid: count / self.num_genotypes
            for rsid, count in self.rsid_counts.items()
        }

    def get_common_rsids(self):
        return {
            rsid
            for rsid, count in self.rsid_counts.items()
            if count == self.num_genotypes
        }

    def save(self, out_path: str | bytes | PathLike[str] | PathLike[bytes]):
        os.makedirs(out_path, exist_ok=True)

    def load(self, data_path: str | bytes | PathLike[str] | PathLike[bytes]):
        for file_name in os.listdir(data_path):
            if file_name.startswith('build_') and file_name.endswith('_rsids.csv'):
                self.build = int(file_name.split('_')[1])
                self.num_genotypes = int(file_name.split('_')[3])
                self.rsids = pd.read_csv(os.path.join(data_path, file_name), index_col=0)
                return
        raise FileNotFoundError('No rsids file found')
