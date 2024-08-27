import ast
from os import PathLike
from typing import Self

import pandas as pd


class GeneMap:
    """
    A class to represent a gene map.

    Attributes
    ----------
    map : pd.DataFrame
        A DataFrame to store the gene map data.

    Methods
    -------
    __len__():
        Returns the number of genes in the map.
    __getitem__(item: str) -> set[int]:
        Returns the set of rs#s for a given gene.
    __iter__():
        Iterates over the gene map.
    from_raw(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        Loads the gene map from a raw file.
    from_processed(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        Loads the gene map from a processed file.
    save(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        Saves the gene map to a file.
    """

    def __init__(self):
        """
        Initializes the GeneMap with an empty map.
        """
        self.map = None

    def __len__(self) -> int:
        """
        Returns the number of genes in the map.

        Returns
        -------
        int
            The number of genes in the map.
        """
        return len(self.map)

    def __getitem__(self, item: str) -> set[int]:
        """
        Returns the set of rs#s for a given gene.

        Parameters
        ----------
        item : str
            The gene name.

        Returns
        -------
        set[int]
            A set of rs#s associated with the gene.
        """
        return set(self.map.loc[item]['rs#s'])

    def __iter__(self):
        """
        Iterates over the gene map.

        Yields
        ------
        tuple
            A tuple containing the gene name and the list of rs#s.
        """
        for index, row in self.map.iterrows():
            yield row['gene'], row['rs#s']

    def from_raw(self,
                 file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        """
        Loads the gene map from a raw file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the raw file.

        Returns
        -------
        Self
            The GeneMap instance.
        """
        columns = ['rs#', 'map wgt', 'snp type', 'chr hits', 'ctg hits', 'total hits', 'chr', 'ctg acc', 'ctg ver',
                   'ctg ID', 'ctg pos', 'chr pos', 'local loci', 'avg het', 's.e. het', 'max prob', 'vali- dated',
                   'geno- types', 'link outs', 'orig build', 'upd build', 'ref- alt', 'sus- pect', 'clin sig.',
                   'allele origin', 'gmaf']
        df = pd.read_csv(file_path, sep='\t', skiprows=7, header=None)
        df.columns = columns
        df = df[['rs#', 'local loci']]
        df = df.rename(columns={'local loci': 'gene'})
        df = df[~df['gene'].isna()]
        df['rs#'] = df['rs#'].astype(int)
        df = df.assign(gene=df['gene'].str.split(',')).explode('gene')
        df = df.groupby('gene')['rs#'].apply(list).reset_index()
        df.columns = ['gene', 'rs#s']
        df['rs#s'] = df['rs#s'].apply(sorted)
        df = df.set_index('gene')
        self.map = df
        return self

    def from_processed(self,
                       file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        """
        Loads the gene map from a processed file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the processed file.

        Returns
        -------
        Self
            The GeneMap instance.
        """
        df = pd.read_csv(file_path, sep='\t', index_col='gene')
        df['rs#s'] = df['rs#s'].apply(ast.literal_eval)
        self.map = df
        return self

    def save(self,
             file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Saves the gene map to a file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the file where the gene map will be saved.
        """
        self.map.save(file_path, sep='\t')
