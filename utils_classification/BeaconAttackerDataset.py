import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset


class BeaconAttackerDataset(Dataset):
    """
    A custom Dataset class for Beacon Attacker.

    This dataset class handles the target genomes, pool presences, and reference frequencies,
    and prepares the data for use in a PyTorch DataLoader.

    Attributes:
        data (torch.Tensor): The concatenated data tensor containing target genomes, pool presences, and reference frequencies.
        labels (torch.Tensor): The labels tensor.
    """

    def __init__(self,
                 target_genomes: npt.NDArray[np.bool_],
                 pool_presences: npt.NDArray[np.bool_],
                 reference_frequencies: npt.NDArray[np.float64],
                 labels: npt.NDArray[np.bool_],
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the BeaconAttackerDataset.

        Args:
            target_genomes (npt.NDArray[np.bool_]): An array of target genomes.
            pool_presences (npt.NDArray[np.bool_]): An array of pool presences.
            reference_frequencies (npt.NDArray[np.float64]): An array of reference frequencies.
            labels (npt.NDArray[np.bool_]): An array of labels.
            dtype (torch.dtype, optional): The data type for the tensors. Defaults to torch.float32.
        """
        num_genomes = target_genomes.shape[0]
        num_snps = target_genomes.shape[1]
        target_genomes = target_genomes[..., np.newaxis]
        pool_presences = np.broadcast_to(pool_presences, (num_genomes, num_snps))[..., np.newaxis]
        reference_frequencies = np.broadcast_to(reference_frequencies, (num_genomes, num_snps))[..., np.newaxis]
        data = np.concatenate((target_genomes, pool_presences, reference_frequencies), axis=2)
        self.data = torch.tensor(data, dtype=dtype)
        self.labels = torch.tensor(labels, dtype=dtype)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self,
                    item: int | slice | list[bool | int] | npt.NDArray[np.bool_ | np.int_]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample and its label from the dataset.

        Args:
            item (int | slice | list[bool | int] | npt.NDArray[np.bool_ | np.int_]): The index or indices of the sample(s) to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The data and label tensors for the specified sample(s).
        """
        return self.data[item], self.labels[item]