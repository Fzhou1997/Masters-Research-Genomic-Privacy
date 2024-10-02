import math

import torch
from torch.utils.data import Subset

from .LSTMAttackerDataset import LSTMAttackerDataset


def stratified_random_split(
        dataset: LSTMAttackerDataset,
        lengths: list[int] | list[float]) -> list[Subset]:
    """
    Perform a stratified random split on the given dataset.

    This function splits the dataset into subsets based on the specified lengths,
    ensuring that each subset contains a proportional representation of each class.

    Args:
        dataset (LSTMAttackerDataset): The dataset to be split.
        lengths (list[int] | list[float]): A list of lengths or proportions for each class.

    Returns:
        list[Subset]: A list of subsets of the dataset.

    Raises:
        ValueError: If the number of classes does not match the number of lengths.
        ValueError: If the sum of lengths is not 1 or equal to the number of samples.
    """
    num_samples = len(dataset)
    num_classes = len(torch.unique(dataset.targets))
    if not math.isclose(sum(lengths), 1) and sum(lengths) != num_samples:
        raise ValueError("The sum of lengths must be 1 or equal to the number of samples.")
    if all(isinstance(length, float) for length in lengths):
        lengths = [int(length * num_samples) for length in lengths]
        if sum(lengths) < num_samples:
            lengths[-1] += num_samples - sum(lengths)
    indices = [torch.where(dataset.targets == i)[0] for i in range(num_classes)]
    subsets = []
    for i, length in enumerate(lengths):

        subset_indices = perms[i][:length]
        subsets.append(Subset(dataset, subset_indices))
        perms[i] = perms[i][length:]
    return subsets


