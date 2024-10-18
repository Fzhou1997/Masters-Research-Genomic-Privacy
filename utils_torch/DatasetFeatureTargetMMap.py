from mmap import mmap

import numpy as np
from torch import Tensor, Dataset


class DatasetFeatureTargetMMap(Dataset):

    features_mmap: mmap
    targets_mmap: mmap
    shape: tuple
    dtype: np.dtype

    def __init__(self,
                 features_mmap: mmap,
                 targets_mmap: mmap,
                 shape: tuple,
                 dtype: np.dtype):
        self.features_mmap = features_mmap
        self.targets_mmap = targets_mmap
        self.shape = shape
        self.dtype = dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:

