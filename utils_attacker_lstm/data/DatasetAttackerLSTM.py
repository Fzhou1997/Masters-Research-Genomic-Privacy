import numpy as np
import numpy.typing as npt
import torch
from torch import tensor

from utils_torch.data import DatasetFeatureTargetClassificationBinarySequential


class DatasetAttackerLSTM(DatasetFeatureTargetClassificationBinarySequential):
    """
    An abstract base class for attacker datasets.

    This class defines the basic structure of an attacker dataset, which includes data and targets tensors.

    Attributes:
        features (torch.Tensor): The features tensor.
        targets (torch.Tensor): The targets tensor.
    """

    def __init__(self,
                 features: npt.NDArray[np.float64],
                 targets: npt.NDArray[np.bool_],
                 dtype: torch.dtype = torch.float32) -> None:
        """
        Initialize the AbstractAttackerDataset.

        Args:
            features (npt.NDArray[np.float64]): The data array.
            targets (npt.NDArray[np.bool_]): The labels array.
            dtype (torch.dtype, optional): The data type for the tensors. Defaults to torch.float32.
        """
        self.features = tensor(features, dtype=dtype)
        self.targets = tensor(targets, dtype=dtype)

    @property
    def num_genomes(self) -> int:
        """
        Returns the number of genomes in the dataset.

        Returns:
            int: Number of genomes.
        """
        return self.num_samples

    @property
    def num_snps(self) -> int:
        """
        Returns the number of SNPs in the dataset.

        Returns:
            int: Number of SNPs.
        """
        return self.num_timesteps

    @property
    def num_features(self) -> int:
        """
        Returns the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return self.features.shape[2]
