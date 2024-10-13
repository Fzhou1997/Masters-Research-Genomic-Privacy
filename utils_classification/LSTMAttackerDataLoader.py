import math
import random

from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .LSTMAttackerDataset import LSTMAttackerDataset


class LSTMAttackerDataLoader(DataLoader):
    def __init__(self,
                 dataset: LSTMAttackerDataset | Subset,
                 genome_batch_size: int = 32,
                 snp_batch_size: int = 4096,
                 shuffle: bool = False,
                 **kwargs):
        super().__init__(dataset, batch_size=genome_batch_size, shuffle=shuffle, **kwargs)
        self.sample_batch_size = genome_batch_size
        self.snp_batch_size = snp_batch_size
        self.sample_indices = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.sample_indices)

    @property
    def num_genomes(self) -> int:
        return len(self.dataset)

    @property
    def num_snps(self) -> int:
        if isinstance(self.dataset, Subset):
            return self.dataset.dataset.shape[1]
        return self.dataset.shape[1]

    @property
    def num_features(self) -> int:
        if isinstance(self.dataset, Subset):
            return self.dataset.dataset.shape[2]
        return self.dataset.shape[2]

    @property
    def num_genome_batches(self) -> int:
        return math.ceil(self.num_genomes / self.sample_batch_size)

    @property
    def num_snp_batches(self) -> int:
        return math.ceil(self.num_snps / self.snp_batch_size)

    @property
    def num_batches(self) -> tuple[int, int]:
        return self.num_genome_batches, self.num_snp_batches

    def get_data_batch(self,
                       genome_batch_index: int,
                       snp_batch_index: int) -> Tensor:
        if genome_batch_index >= self.sample_batch_size:
            raise IndexError('Sample batch index out of range.')
        if snp_batch_index >= self.snp_batch_size:
            raise IndexError('SNP batch index out of range.')
        sample_start = genome_batch_index * self.sample_batch_size
        sample_end = min(sample_start + self.sample_batch_size, self.num_genomes)
        snp_start = snp_batch_index * self.snp_batch_size
        snp_end = min(snp_start + self.snp_batch_size, self.num_snps)
        return self.dataset.data[self.sample_indices[sample_start:sample_end], snp_start:snp_end, :]

    def get_target_batch(self, sample_batch_index: int) -> Tensor:
        if sample_batch_index >= self.sample_batch_size:
            raise IndexError('Sample batch index out of range.')
        sample_start = sample_batch_index * self.sample_batch_size
        sample_end = min(sample_start + self.sample_batch_size, self.num_genomes)
        return self.dataset.targets[self.sample_indices[sample_start:sample_end]]
