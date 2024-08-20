import json
from os import PathLike, makedirs
from typing import Self

from pandas import DataFrame, Index, concat
from torch import Tensor, tensor

from utils_genomes import Genotype


class Genomes:
    def __init__(self, build):
        self.build = build
        self.num_genotypes = 0
        self.genotypes = DataFrame()
        self.phenotypes = DataFrame()

    def _to_dict(self) -> dict[str, int | dict[str, dict[str, str]]]:
        return {
            'build': self.build,
            'num_genotypes': self.num_genotypes,
            'genotypes': self.genotypes.to_dict(orient='index'),
            'phenotypes': self.phenotypes.to_dict(orient='index'),
        }

    def _from_dict(self, data: dict[str, int | dict[str, dict[str, str]]]) -> Self:
        self.build = data['build']
        self.num_genotypes = data['num_genotypes']
        self.genotypes = DataFrame.from_dict(data['genotypes'], orient='index')
        self.phenotypes = DataFrame.from_dict(data['phenotypes'], orient='index')
        return self

    def concat_genotype(self, genotype: Genotype) -> None:
        if genotype.get_build() != self.build:
            raise ValueError('Reference genome build mismatch')
        self.num_genotypes += 1
        user_id = genotype.get_user_id()
        user_genotype = genotype.get_genotype().T
        user_genotype.index = Index([user_id], name='user_id')
        self.genotypes = concat([self.genotypes, user_genotype], ignore_index=False)

    def concat_phenotypes(self, phenotypes) -> None:
        self.phenotypes = concat([self.phenotypes, phenotypes.get_phenotypes()], ignore_index=False)

    def filter_genotypes_na(self) -> None:
        self.genotypes = self.genotypes.dropna(axis='columns')

    def filter_phenotypes_na(self) -> None:
        self.phenotypes = self.phenotypes.dropna(axis='rows')

    def filter_phenotypes_genotypes(self) -> None:
        self.phenotypes = self.phenotypes.loc[self.genotypes.index]

    def get_build(self) -> int:
        return self.build

    def get_num_genotypes(self) -> int:
        return self.num_genotypes

    def get_user_ids(self) -> list[int]:
        return list(self.genotypes.index)

    def get_rsids(self) -> list[str]:
        return list(self.genotypes.columns)

    def get_genotypes(self) -> DataFrame:
        return self.genotypes

    def get_phenotypes(self) -> DataFrame:
        return self.phenotypes

    def get_genomes(self) -> DataFrame:
        return concat([self.genotypes, self.phenotypes], axis=1, ignore_index=False)

    def get_genotypes_tensor(self) -> Tensor:
        genotypes_values = self.genotypes.values
        return tensor(genotypes_values)

    def get_phenotypes_tensor(self) -> Tensor:
        phenotypes_values = self.phenotypes.values
        return tensor(phenotypes_values).squeeze()

    def get_tensor(self) -> Tensor:
        genomes_values = self.get_genomes().values
        return tensor(genomes_values)

    def sort_rsids(self, rsids: list[str]) -> None:
        self.genotypes = self.genotypes[rsids]

    def save(self, out_path: str | bytes | PathLike[str] | PathLike[bytes], file_name: str) -> None:
        makedirs(out_path, exist_ok=True)
        with open(f'{out_path}/{file_name}.json', 'w') as file:
            json.dump(self._to_dict(), file)

    def load(self, in_path: str | bytes | PathLike[str] | PathLike[bytes], file_name: str) -> Self:
        with open(f'{in_path}/{file_name}.json', 'r') as file:
            return self._from_dict(json.load(file))
