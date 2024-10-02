import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, tensor
from torch.utils.data import Subset


def stratified_random_split(
        dataset: Tensor,
        lengths: list[float],
        seed: int = 42) -> list[Subset]:
    classes = torch.unique(dataset.targets)
    num_samples = len(dataset)
    num_subsets = len(lengths)
    num_classes = len(classes)
    class_indices = [torch.where(dataset.targets == c)[0] for c in classes]
    num_samples_by_class = tensor([len(indices) for indices in class_indices], dtype=torch.float)
    fraction_by_subset = tensor(lengths, dtype=torch.float)
    num_samples_by_class_by_subset = torch.floor(num_samples_by_class.reshape((num_classes, 1)) @ fraction_by_subset.reshape((1, num_subsets))).int()
    remainder_by_class = num_samples_by_class - torch.sum(num_samples_by_class_by_subset, dim=1)
    remainder_by_class_per_subset = (remainder_by_class // num_subsets).int()
    num_samples_by_class_by_subset += remainder_by_class_per_subset.reshape((num_classes, 1))
    remainder_by_class_per_subset = (remainder_by_class % num_subsets).int()
    range = torch.arange(num_subsets)
    mask = (range.reshape((1, num_subsets)) < remainder_by_class_per_subset.reshape((num_classes, 1))).int()
    num_samples_by_class_by_subset += mask
    subsets = []
