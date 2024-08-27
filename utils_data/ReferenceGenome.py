from os import PathLike
from typing import Self

from Bio import SeqIO


class ReferenceGenome:
    """
    A class to represent a reference genome.

    Attributes
    ----------
    seq : str
        The sequence of the genome.

    Methods
    -------
    __len__():
        Returns the length of the genome sequence.
    __getitem__(item: int):
        Returns the reference allele at the specified position in the genome sequence.
    __iter__():
        Returns an iterator for the genome sequence.
    from_raw(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        Loads the genome sequence from a raw FASTA file.
    from_processed(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        Loads the genome sequence from a processed text file.
    save(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        Saves the genome sequence to a file.
    """

    def __init__(self):
        """
        Initializes the ReferenceGenome with an empty sequence.
        """
        self.seq = None

    def __len__(self) -> int:
        """
        Returns the length of the genome sequence.

        Returns
        -------
        int
            The length of the genome sequence.
        """
        return len(self.seq)

    def __getitem__(self, item: int) -> str:
        """
        Returns the reference allele at the specified position in the genome sequence.
        Positions are 1-based indexed.

        Parameters
        ----------
        item : int
            The position in the genome sequence (1-based index).

        Returns
        -------
        str
            The character at the specified position.
        """
        return self.seq[item - 1]

    def __iter__(self):
        """
        Returns an iterator for the genome sequence.

        Returns
        -------
        iterator
            An iterator for the genome sequence.
        """
        return iter(self.seq)

    def from_raw(self,
                 file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        """
        Loads the genome sequence from a raw FASTA file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the raw FASTA file.

        Returns
        -------
        Self
            The ReferenceGenome instance with the loaded sequence.
        """
        record = SeqIO.read(file_path, 'fasta')
        self.seq = str(record.seq)
        return self

    def from_processed(self,
                       file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        """
        Loads the genome sequence from a processed text file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the processed text file.

        Returns
        -------
        Self
            The ReferenceGenome instance with the loaded sequence.
        """
        with open(file_path, 'r') as f:
            self.seq = f.read()
        return self

    def save(self,
             file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Saves the genome sequence to a file.

        Parameters
        ----------
        file_path : str | bytes | PathLike[str] | PathLike[bytes]
            The path to the file where the sequence will be saved.
        """
        with open(file_path, 'w') as f:
            f.write(self.seq)
