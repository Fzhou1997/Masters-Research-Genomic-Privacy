import json
from collections import Counter
from os import PathLike, makedirs
from typing import Self

from utils_genomes import Genotype


class Rsids:
    def __init__(self, build: int):
        self.build = build
        self.num_genotypes = 0
        self.rsid_counts = Counter()
        self.rsid_map = {}

    def _to_dict(self) -> dict[str, int | dict[str, int] | dict[str, dict[str, int]]]:
        return {
            'build': self.build,
            'num_genotypes': self.num_genotypes,
            'rsid_counts': dict(self.rsid_counts),
            'rsid_map': self.rsid_map,
        }

    def _from_dict(self, data: dict[str, int | dict[str, int] | dict[str, dict[str, int]]]) -> Self:
        self.build = data['build']
        self.num_genotypes = data['num_genotypes']
        self.rsid_counts = Counter(data['rsid_counts'])
        self.rsid_map = data['rsid_map']
        return self

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

    def get_rsid_probabilities(self) -> dict[str, float]:
        return {
            rsid: count / self.num_genotypes
            for rsid, count in self.rsid_counts.items()
        }

    def get_common_rsids(self) -> set[str]:
        return {
            rsid
            for rsid, count in self.rsid_counts.items()
            if count == self.num_genotypes
        }

    def get_sorted_rsids(self) -> list[str]:
        return sorted(self.rsid_map, key=lambda rsid: (self.rsid_map[rsid]['chrom'], self.rsid_map[rsid]['pos']))

    def save(self, out_path: str | bytes | PathLike[str] | PathLike[bytes], file_name: str) -> None:
        makedirs(out_path, exist_ok=True)
        with open(f'{out_path}/{file_name}.json', 'w') as file:
            json.dump(self._to_dict(), file)

    def load(self, in_path: str | bytes | PathLike[str] | PathLike[bytes], file_name: str) -> Self:
        with open(f'{in_path}/{file_name}.json', 'r') as file:
            return self._from_dict(json.load(file))
