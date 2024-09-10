from os import PathLike
from typing import Self

import pandas as pd


class Beacon:
    def __init__(self):
        self.population_gnomes = None

    def __getitem__(self,
                    rsid: int | list[int]) -> bool:
        return self.population_gnomes[rsid].any()

    def from_processed(self,
                       file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        self.population_gnomes = pd.read_csv(file_path, index_col=0)
        return self