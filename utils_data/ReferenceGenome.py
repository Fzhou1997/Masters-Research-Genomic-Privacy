import pickle
from os import PathLike
from typing import Self

from Bio import SeqIO
from bitarray import bitarray

_nucleotide_to_bits = {
    "A": "0000",
    "C": "0001",
    "G": "0010",
    "T": "0011",
    "N": "1111"
}

_bits_to_nucleotide = {
    "0000": "A",
    "0001": "C",
    "0010": "G",
    "0011": "T",
    "1111": "N"
}

_num_bits_per_nucleotide = 4

class ReferenceGenome:
    """
    A class to represent a reference genome.

    Attributes
    ----------
    sequence : str
        The sequence of the genome.

    Methods
    -------
    __len__():
        Returns the length of the genome sequence.
    __getitem__(item: int | slice | list[int]) -> str:
        Returns the reference allele at the specified position in the genome sequence.
    __str__():
        Returns the reference genome sequence as a string.
    __iter__():
        Returns an iterator for the genome sequence.
    from_raw(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        Loads the genome sequence from a raw FASTA file.
    from_processed(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> Self:
        Loads the genome sequence from a processed text file.
    save(file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        Saves the genome sequence to a file.
    """

    sequence: bitarray

    def __init__(self):
        """
        Initializes the ReferenceGenome with an empty sequence.
        """
        self.sequence = bitarray()

    def __len__(self) -> int:
        """
        Returns the length of the genome sequence.

        Returns
        -------
        int
            The length of the genome sequence.
        """
        return len(self.sequence) // _num_bits_per_nucleotide

    def __getitem__(self, idx: int | slice | list[int]) -> str:
        """
        Returns the reference allele at the specified position in the genome sequence.

        Parameters
        ----------
        idx : int | slice | list[int]
            The position(s) in the genome sequence.

        Returns
        -------
        str
            The reference allele(s) at the specified position(s) in the genome sequence.
        """
        if isinstance(idx, int):
            start = (idx - 1) * _num_bits_per_nucleotide
            end = start + _num_bits_per_nucleotide
            bits = self.sequence[start:end].to01()
            return _bits_to_nucleotide[bits]
        elif isinstance(idx, slice):
            start = (idx.start - 1) * _num_bits_per_nucleotide if idx.start else None
            end = idx.stop * _num_bits_per_nucleotide if idx.stop else None
            bits = self.sequence[start:end].to01()
            return "".join(_bits_to_nucleotide[bits[i:i + _num_bits_per_nucleotide]] for i in range(0, len(bits), _num_bits_per_nucleotide))
        elif isinstance(idx, list):
            nucleotides = ""
            for i in idx:
                start = (i - 1) * _num_bits_per_nucleotide
                end = start + _num_bits_per_nucleotide
                bits = self.sequence[start:end].to01()
                nucleotides += _bits_to_nucleotide[bits]
            return nucleotides

    def __str__(self) -> str:
        """
        Returns the genome sequence as a string.

        Returns
        -------
        str
            The genome sequence.
        """
        bits = self.sequence.to01()
        return "".join(_bits_to_nucleotide[bits[i:i + _num_bits_per_nucleotide]] for i in range(0, len(bits), _num_bits_per_nucleotide))

    def __iter__(self):
        """
        Returns an iterator for the genome sequence.

        Yields
        -------
        str
            The reference allele at each position in the genome sequence
        """
        for i in range(0, len(self.sequence), _num_bits_per_nucleotide):
            bits = self.sequence[i:i + _num_bits_per_nucleotide].to01()
            yield _bits_to_nucleotide[bits]

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
        record = str(SeqIO.read(file_path, 'fasta').seq)
        self.sequence = bitarray(''.join(_nucleotide_to_bits[nucleotide] for nucleotide in record))
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
        with open(file_path, 'rb') as f:
            self.sequence = pickle.load(f)
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
        with open(file_path, 'wb') as f:
            pickle.dump(self.sequence, f)
