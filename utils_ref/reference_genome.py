import re
from os import PathLike
from typing import Self

from Bio import SeqIO


class ReferenceGenome:
    def __init__(self):
        self.bld = None
        self.chr = None
        self.seq = None

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, item: int) -> str:
        return self.chr[item]

    def __iter__(self):
        return iter(self.seq)

    def from_fasta(self, file_path: str | bytes | PathLike[str] | PathLike[bytes], file_name: str) -> Self:
        record = SeqIO.read(f'{file_path}/{file_name}', 'fasta')
        build_match = re.search(r'(?<=GRCh)\d+(?=:)', record.description)
        chr_match = re.search(r'\d+(?= dna:)', record.description)
        self.bld = int(build_match[0])
        self.chr = int(chr_match[0])
        self.seq = str(record.seq)
        return self

    def get_build(self) -> int:
        return self.bld

    def get_chromosome(self) -> int:
        return self.chr

    def get_sequence(self) -> str:
        return self.seq
