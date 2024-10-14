from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, tensor, Size
from torch.utils.data import Dataset

class LSTMAttackerDataset(Dataset):
    """
    An abstract base class for attacker datasets.

    This class defines the basic structure of an attacker dataset, which includes data and targets tensors.

    Attributes:
        data (torch.Tensor): The data tensor.
        targets (torch.Tensor): The targets tensor.
    """

    @abstractmethod
    def __init__(self,
                 data: npt.NDArray[np.float64],
                 labels: npt.NDArray[np.bool_],
                 dtype: torch.dtype = torch.float32) -> None:
        """
        Initialize the AbstractAttackerDataset.

        Args:
            data (npt.NDArray[np.float64]): The data array.
            labels (npt.NDArray[np.bool_]): The labels array.
            dtype (torch.dtype, optional): The data type for the tensors. Defaults to torch.float32.
        """
        self.data = tensor(data, dtype=dtype)
        self.targets = tensor(labels, dtype=dtype)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.data.shape[0]

    def __getitem__(self,
                    item: int | slice | list[bool | int]) -> tuple[Tensor, Tensor]:
        """
        Retrieve a sample and its label from the dataset.

        Args:
            item (int | slice | list[bool | int]): The index or indices of the sample(s) to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The data and label tensors for the specified sample(s).
        """
        return self.data[item], self.targets[item]

    @property
    def shape(self) -> Size:
        """
        Return the shape of the dataset.

        Returns:
            Size: The shape of the dataset.
        """
        return self.data.shape

    @property
    def num_genomes(self) -> int:
        """
        Returns the number of genomes in the dataset.

        Returns:
            int: Number of genomes.
        """
        return self.data.shape[0]

    @property
    def num_snps(self) -> int:
        """
        Returns the number of SNPs in the dataset.

        Returns:
            int: Number of SNPs.
        """
        return self.data.shape[1]

    @property
    def num_features(self) -> int:
        """
        Returns the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return self.data.shape[2]
