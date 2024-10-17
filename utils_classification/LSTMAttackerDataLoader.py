import math
import random

from torch import Tensor
from torch.utils.data import DataLoader, Subset

from utils_torch.FeatureTargetSubset import FeatureTargetSubset
from .LSTMAttackerDataset import LSTMAttackerDataset


class LSTMAttackerDataLoader(DataLoader):
    """
    A custom DataLoader for LSTMAttackerDataset that handles batching for genomes and SNPs.

    Attributes:
        genome_batch_size (int): The batch size for genome samples.
        snp_batch_size (int): The batch size for SNPs.
        sample_indices (list): List of indices for the samples in the dataset.
    """

    dataset: LSTMAttackerDataset | FeatureTargetSubset
    genome_batch_size: int
    snp_batch_size: int
    sample_indices: list[int]

    def __init__(self,
                 dataset: LSTMAttackerDataset | FeatureTargetSubset,
                 genome_batch_size: int = 32,
                 snp_batch_size: int = 4096,
                 shuffle: bool = False,
                 **kwargs):
        """
        Initializes the LSTMAttackerDataLoader.

        Args:
            dataset (LSTMAttackerDataset | Subset): The dataset to load data from.
            genome_batch_size (int, optional): The batch size for genome samples. Defaults to 32.
            snp_batch_size (int, optional): The batch size for SNPs. Defaults to 4096.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            **kwargs: Additional arguments for the DataLoader.
        """
        super().__init__(dataset, batch_size=genome_batch_size, shuffle=shuffle, **kwargs)
        self.genome_batch_size = genome_batch_size
        self.snp_batch_size = snp_batch_size
        self.sample_indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.sample_indices)

    @property
    def num_genomes(self) -> int:
        """
        Returns the number of genomes in the dataset.

        Returns:
            int: Number of genomes.
        """
        return len(self.dataset)

    @property
    def num_snps(self) -> int:
        """
        Returns the number of SNPs in the dataset.

        Returns:
            int: Number of SNPs.
        """
        return self.dataset.shape[1]

    @property
    def num_features(self) -> int:
        """
        Returns the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return self.dataset.shape[2]

    @property
    def num_genome_batches(self) -> int:
        """
        Returns the number of genome batches.

        Returns:
            int: Number of genome batches.
        """
        return math.ceil(self.num_genomes / self.genome_batch_size)

    @property
    def num_snp_batches(self) -> int:
        """
        Returns the number of SNP batches.

        Returns:
            int: Number of SNP batches.
        """
        return math.ceil(self.num_snps / self.snp_batch_size)

    @property
    def num_batches(self) -> tuple[int, int]:
        """
        Returns the number of genome and SNP batches as a tuple.

        Returns:
            tuple[int, int]: Number of genome batches and SNP batches.
        """
        return self.num_genome_batches, self.num_snp_batches

    def get_data_batch(self,
                       genome_batch_index: int,
                       snp_batch_index: int) -> Tensor:
        """
        Retrieves a batch of data from the dataset.

        Args:
            genome_batch_index (int): The index of the genome batch.
            snp_batch_index (int): The index of the SNP batch.

        Returns:
            Tensor: A batch of data from the dataset.

        Raises:
            IndexError: If the genome_batch_index or snp_batch_index is out of range.
        """
        if genome_batch_index >= self.num_genome_batches:
            raise IndexError('Sample batch index out of range.')
        if snp_batch_index >= self.num_snp_batches:
            raise IndexError('SNP batch index out of range.')
        sample_start = genome_batch_index * self.genome_batch_size
        sample_end = min(sample_start + self.genome_batch_size, self.num_genomes)
        snp_start = snp_batch_index * self.snp_batch_size
        snp_end = min(snp_start + self.snp_batch_size, self.num_snps)
        idx = self.sample_indices[sample_start:sample_end], slice(snp_start, snp_end, 1), slice(None, None, 1)
        return self.dataset.get_features(idx)

    def get_target_batch(self, sample_batch_index: int) -> Tensor:
        """
        Retrieves a batch of targets from the dataset.

        Args:
            sample_batch_index (int): The index of the sample batch.

        Returns:
            Tensor: A batch of targets from the dataset.

        Raises:
            IndexError: If the sample_batch_index is out of range.
        """
        if sample_batch_index >= self.num_genome_batches:
            raise IndexError('Sample batch index out of range.')
        sample_start = sample_batch_index * self.genome_batch_size
        sample_end = min(sample_start + self.genome_batch_size, self.num_genomes)
        return self.dataset.get_targets(self.sample_indices[sample_start:sample_end])
