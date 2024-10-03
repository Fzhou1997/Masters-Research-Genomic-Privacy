import torch
from torch.utils.data import DataLoader, Subset

from .LSTMAttackerDataset import LSTMAttackerDataset


class LSTMAttackerDataLoader(DataLoader):
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

    def __iter__(self):
        self.current_index = (0, 0)
        self.sample_indices = torch.randperm(len(self.dataset)) if self.shuffle else self.sample_indices = torch.arange(len(self.dataset))
        return self

    def __next__(self):
        if self.current_index[0] >= len(self.dataset):
            raise StopIteration

