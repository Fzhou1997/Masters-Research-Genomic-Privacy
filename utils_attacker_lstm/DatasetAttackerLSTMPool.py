import numpy as np
import numpy.typing as npt
import torch

from .DatasetAttackerLSTM import DatasetAttackerLSTM


class DatasetAttackerLSTMPool(DatasetAttackerLSTM):
    """
    A custom Dataset class for Pool Attacker.

    This dataset class handles the target genomes, pool frequencies, and reference frequencies,
    and prepares the data for use in a PyTorch DataLoader.

    Attributes:
        features (torch.Tensor): The concatenated data tensor containing target genomes, pool frequencies, and reference frequencies.
        targets (torch.Tensor): The targets tensor.
    """

    def __init__(self,
                 target_genomes: npt.NDArray[np.bool_],
                 pool_frequencies: npt.NDArray[np.float64],
                 reference_frequencies: npt.NDArray[np.float64],
                 labels: npt.NDArray[np.bool_],
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the PoolAttackerDataset.

        Args:
            target_genomes (npt.NDArray[np.bool_]): An array of target genomes.
            pool_frequencies (npt.NDArray[np.float64]): An array of pool frequencies.
            reference_frequencies (npt.NDArray[np.float64]): An array of reference frequencies.
            labels (npt.NDArray[np.bool_]): An array of labels.
            dtype (torch.dtype, optional): The data type for the tensors. Defaults to torch.float32.
        """
        num_genomes = target_genomes.shape[0]
        num_snps = target_genomes.shape[1]
        target_genomes = target_genomes[..., np.newaxis]
        pool_frequencies = np.broadcast_to(pool_frequencies, (num_genomes, num_snps))[..., np.newaxis]
        reference_frequencies = np.broadcast_to(reference_frequencies, (num_genomes, num_snps))[..., np.newaxis]
        data = np.concatenate((target_genomes, pool_frequencies, reference_frequencies), axis=2)
        super().__init__(data, labels, dtype)
