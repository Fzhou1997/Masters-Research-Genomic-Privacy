from collections import Counter
from os import PathLike

import pandas as pd


class SnpCounter:
    """
    A class to count SNP occurrences across multiple genomes.
    """

    def __init__(self):
        """
        Initializes a new SnpCounter instance.
        """
        self.num_genomes = 0
        self.counter = Counter()

    def __len__(self) -> int:
        """
        Returns the number of unique SNPs counted.

        Returns:
            int: The number of unique SNPs.
        """
        return len(self.counter)

    def __getitem__(self, item: int) -> int:
        """
        Returns the count of a specific SNP.

        Args:
            item (int): The SNP identifier.

        Returns:
            int: The count of the specified SNP.
        """
        return self.counter[item]

    def __iter__(self):
        """
        Returns an iterator over the SNPs.

        Returns:
            Iterator: An iterator over the SNPs.
        """
        return iter(self.counter)

    def load(self,
             file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Loads SNP counts from a CSV file.

        Args:
            file_path (str | bytes | PathLike[str] | PathLike[bytes]): The path to the CSV file.
        """
        with open(file_path, 'r') as f:
            header = f.readline()
            self.num_genomes = int(header.split(':')[1])
            counter_df = pd.read_csv(f)
            self.counter = Counter(dict(zip(counter_df['rs#'], counter_df['count'])))

    def save(self,
             file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
        """
        Saves the SNP counts to a CSV file.

        Args:
            file_path (str | bytes | PathLike[str] | PathLike[bytes]): The path to the CSV file.
        """
        header = f'# num_gnomes:{self.num_genomes}\n'
        counter_dict = dict(self.counter)
        counter_df = pd.DataFrame(counter_dict.items(), columns=['rs#', 'count'])
        counter_df = counter_df.sort_values(by='rs#')
        with open(file_path, 'w', newline='') as f:
            f.write(header)
            counter_df.to_csv(f, index=False)

    def update(self, snps: set[int]) -> None:
        """
        Updates the SNP counts with a new set of SNPs from a genome.

        Args:
            snps (set[int]): A set of SNP identifiers.
        """
        self.num_genomes += 1
        self.counter.update(snps)

    def get_num_genomes(self) -> int:
        """
        Returns the number of genomes processed.

        Returns:
            int: The number of genomes.
        """
        return self.num_genomes

    def get_num_snps_above_threshold(self,
                                     threshold: float) -> int:
        """
        Returns the number of SNPs that appear in at least the given fraction of genomes.

        Args:
            threshold (float): The fraction of genomes a SNP must appear in to be counted.

        Returns:
            int: The number of SNPs above the threshold.
        """
        min_count = threshold * self.num_genomes
        return sum(count >= min_count for count in self.counter.values())

    def get_snps_above_threshold(self,
                                 threshold: float) -> set[int]:
        """
        Returns the set of SNPs that appear in at least the given fraction of genomes.

        Args:
            threshold (float): The fraction of genomes a SNP must appear in to be included.

        Returns:
            set[int]: The set of SNPs above the threshold.
        """
        min_count = threshold * self.num_genomes
        return set([snp for snp, count in self.counter.items() if count >= min_count])
