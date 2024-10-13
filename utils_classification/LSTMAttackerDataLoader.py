import torch
from torch.utils.data import DataLoader, Subset

from .LSTMAttackerDataset import LSTMAttackerDataset


class LSTMAttackerDataLoader:
    def __init__(self,
                 dataset: LSTMAttackerDataset | Subset,
                 batch_size: tuple[int, int],
                 shuffle: bool = False,
                 weighted_sampling: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weighted_sampling = weighted_sampling

        self.current_index = None
        self.sample_indices = None

    @property
    def num_batches(self) -> tuple[int, int]:
        num_samples = self.dataset.shape[0]
        num_snps = self.dataset.shape[1]
        num_samples_per_batch = self.batch_size[0]
        num_snps_per_batch = self.batch_size[1]
        return num_samples // num_samples_per_batch, num_snps // num_snps_per_batch


