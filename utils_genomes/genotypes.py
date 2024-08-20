import itertools
import json
from os import PathLike, makedirs
from typing import Self

import pandas as pd

from utils_genomes import Genotype

ALLELES = 'ACGTDI-'
GENOTYPES = [''.join(item) for item in itertools.product('ACGT', repeat=2)] + [''.join(item) for item in itertools.product('DI', repeat=2)] + ['--']
TRANSFORMATION = pd.DataFrame(columns=list(ALLELES), index=GENOTYPES)
for genotype in GENOTYPES:
    for allele in ALLELES:
        TRANSFORMATION.at[genotype, allele] = genotype.count(allele)


class Genotypes:
    def __init__(self, build: int):
        self.build = build
        self.num_genotypes = 0
        self.genotype_counts = {}

    def _to_dict(self) -> dict[str, int | dict[str, dict[str, int]]]:
        return {
            'build': self.build,
            'num_genotypes': self.num_genotypes,
            'genotype_counts': self.genotype_counts,
        }

    def _from_dict(self, data: dict[str, int | dict[str, dict[str, int]]]) -> Self:
        self.build = data['build']
        self.num_genotypes = data['num_genotypes']
        self.genotype_counts = data['genotype_counts']
        return self

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.genotype_counts).T

    def _from_dataframe(self, data: pd.DataFrame) -> Self:
        self.genotype_counts = data.to_dict(orient='index')
        return self

    def concat_genotype(self, genotype: Genotype) -> None:
        if genotype.get_build() != self.build:
            raise ValueError('Reference genome build mismatch')
        self.num_genotypes += 1
        for rsid, _genotype in genotype:
            if rsid not in self.genotype_counts:
                self.genotype_counts[rsid] = {__genotype: 0 for __genotype in GENOTYPES}
            self.genotype_counts[rsid][_genotype] += 1

    def concat_genotypes(self, genotypes: list[Genotype]) -> None:
        for genotype in genotypes:
            self.concat_genotype(genotype)

    def get_build(self) -> int:
        return self.build

    def get_num_genotypes(self) -> int:
        return self.num_genotypes

    def get_genotype_counts(self) -> dict[str, dict[str, int]]:
        return self.genotype_counts

    def get_genotype_probabilities(self) -> dict[str, dict[str, float]]:
        count_df = self._to_dataframe()
        return count_df.div(count_df.sum(axis=1), axis=0).to_dict(orient='index')

    def get_mode_genotypes(self) -> dict[str, str]:
        count_df = self._to_dataframe()
        count_df = count_df.drop(columns=['--'])
        return count_df.idxmax(axis=1).to_dict()

    def get_allele_counts(self) -> dict[str, dict[str, int]]:
        count_df = self._to_dataframe()
        return count_df.dot(TRANSFORMATION).to_dict(orient='index')

    def get_allele_probabilities(self) -> dict[str, dict[str, float]]:
        count_df = self._to_dataframe()
        return count_df.dot(TRANSFORMATION).div(count_df.sum(axis=1), axis=0).to_dict(orient='index')

    def get_reference_alleles(self) -> dict[str, str]:
        count_df = self._to_dataframe()
        return count_df.dot(TRANSFORMATION).idxmax(axis=1).to_dict()

    def save(self, out_path: str | bytes | PathLike[str] | PathLike[bytes], file_name: str) -> None:
        makedirs(out_path, exist_ok=True)
        with open(f'{out_path}/{file_name}.json', 'w') as file:
            json.dump(self._to_dict(), file)

    def load(self, in_path: str | bytes | PathLike[str] | PathLike[bytes], file_name: str) -> Self:
        with open(f'{in_path}/{file_name}.json', 'r') as file:
            return self._from_dict(json.load(file))
