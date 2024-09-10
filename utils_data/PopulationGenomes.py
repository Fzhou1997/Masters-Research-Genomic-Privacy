from os import PathLike
from typing import Self

import pandas as pd


class PopulationGenomes:
    def __init__(self):
        self.genomes = None
        self.num_genomes = None

    def from_processed_individual_genomes(self,
                                          file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:

        return self