from os import PathLike
from typing import Self

import pandas as pd

_gen_header = ["chr", "rsid", "pos", "ref", "alt"]
_gen_delimiter = " "

class PopulationGenomes:

    chromosome: int
    num_genomes: int

    def __init__(self):
        self.genomes = None
        self.num_genomes = None

    def to_gen(self, file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        pass