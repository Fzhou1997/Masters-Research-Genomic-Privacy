import re
from os import PathLike
from typing import Self

from pandas import read_csv, DataFrame


class SnpMap:
    def __init__(self):
        self.map = None

    def __len__(self):
        return len(self.map)

    def __getitem__(self, item: int):
        row = self.map.loc[item]
        return row['chr'], row['pos']

    def __iter__(self):
        for rs, row in self.map.iterrows():
            yield rs, row['chr'], row['pos']

    def from_raw(self,
                 file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        columns = ['rs#', 'map wgt', 'snp type', 'chr hits', 'ctg hits', 'total hits', 'chr', 'ctg acc', 'ctg ver', 'ctg ID', 'ctg pos', 'chr pos', 'local loci', 'avg het', 's.e. het', 'max prob', 'vali- dated', 'geno- types', 'link outs', 'orig build', 'upd build', 'ref- alt', 'sus- pect', 'clin sig.', 'allele origin', 'gmaf']
        df = read_csv(file_path, sep='\t', skiprows=7, header=None)
        df.columns = columns
        df = df[['rs#', 'chr', 'chr pos']]
        df = df.rename(columns={'chr pos': 'pos'})
        df = df[df['pos'] != ' ']
        df['rs#'] = df['rs#'].astype(int)
        df['chr'] = df['chr'].astype(int)
        df['pos'] = df['pos'].astype(int)
        df = df.set_index('rs#')
        self.map = df.squeeze()
        return self

    def from_processed(self,
                       file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        self.map = read_csv(file_path, sep='\t', index_col='rs#')
        return self

    def save(self,
             file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        self.map.to_csv(file_path, sep='\t')
