from os import PathLike
from typing import Self

from pandas import read_csv, DataFrame


class SnpMap:
    """
    A class to represent a SNP (Single Nucleotide Polymorphism) map.

    Attributes
    ----------
    map : pandas.DataFrame
        A DataFrame to store SNP data.

    Methods
    -------
    __len__():
        Returns the number of SNPs in the map.
    __getitem__(item: int):
        Returns the chromosome and position of the SNP at the given rs number.
    __iter__():
        Iterates over the SNP map, yielding the rs number, chromosome, and position.
    from_raw(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        Loads SNP data from a raw SNP map file.
    from_processed(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        Loads SNP data from a processed SNP map file.
    save(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        Saves the SNP map to a file.
    """

    map: DataFrame

    def __init__(self):
        """Initializes the SnpMap with an empty map."""
        self.map = None

    def __len__(self):
        """
        Returns the number of SNPs in the map.

        Returns
        -------
        int
            The number of SNPs.
        """
        return len(self.map)

    def __getitem__(self, item: int):
        """
        Returns the chromosome and position of the SNP with the given rs number.

        Parameters
        ----------
        item : int
            The rs number of the SNP.

        Returns
        -------
        tuple
            A tuple containing the chromosome and position of the SNP.
        """
        row = self.map.loc[item]
        return row['chr'], row['pos']

    def __iter__(self):
        """
        Iterates over the SNP map, yielding the rs number, chromosome, and position.

        Yields
        ------
        tuple
            A tuple containing the rs number, chromosome, and position of each SNP.
        """
        for rs, row in self.map.iterrows():
            yield rs, row['chr'], row['pos']

    def sort(self) -> Self:
        self.map = self.map.sort_values(by=['chr', 'pos'])
        return self

    def from_raw(self,
                 file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        """
        Loads SNP data from a raw SNP map file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the raw SNP map file.

        Returns
        -------
        Self
            The SnpMap instance with loaded data.
        """
        columns = ['rs#', 'map wgt', 'snp type', 'chr hits', 'ctg hits', 'total hits', 'chr', 'ctg acc', 'ctg ver',
                   'ctg ID', 'ctg pos', 'chr pos', 'local loci', 'avg het', 's.e. het', 'max prob', 'vali- dated',
                   'geno- types', 'link outs', 'orig build', 'upd build', 'ref- alt', 'sus- pect', 'clin sig.',
                   'allele origin', 'gmaf']
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
        """
        Loads SNP data from a processed SNP map file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the processed SNP data file.

        Returns
        -------
        Self
            The SnpMap instance with loaded data.
        """
        self.map = read_csv(file_path, sep='\t', index_col='rs#')
        return self

    def save(self,
             file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Saves the SNP map to a file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the file where the SNP map will be saved.
        """
        self.map.to_csv(file_path, sep='\t')
