import random
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset


def stratified_random_split(
        dataset: Dataset,
        ratios: list[float]) -> list[Subset]:
    """
    Splits a dataset into stratified random subsets based on the provided ratios.

    Args:
        dataset (Dataset): The dataset to split. Must have a 'targets' attribute.
        ratios (List[float]): The ratios for splitting the dataset.

    Returns:
        List[Subset]: A list of Subset objects corresponding to the splits.
    """
    if not hasattr(dataset, 'targets'):
        raise AttributeError('Dataset must have a targets attribute.')
    if not isinstance(dataset.targets, Tensor):
        raise TypeError('Dataset targets must be a torch.Tensor.')
    classes = torch.unique(dataset.targets)
    num_subsets = len(ratios)
    subsets_indices = [[] for _ in range(num_subsets)]
    for c in classes:
        class_indices = torch.where(dataset.targets == c)[0].tolist()
        class_subsets_indices = _random_split(class_indices, ratios)
        for s in range(num_subsets):
            subsets_indices[s].extend(class_subsets_indices[s])
    for s in range(num_subsets):
        random.shuffle(subsets_indices[s])
    subsets = [Subset(dataset, subset_indices) for subset_indices in subsets_indices]
    return subsets

def _random_split(
        data: list[Any],
        ratios: list[float]) -> list[list[Any]]:
    """
    Splits data into random subsets based on the provided ratios.

    Args:
        data (List[Any]): The data to split.
        ratios (List[float]): The ratios for splitting the data.

    Returns:
        List[List[Any]]: A list of lists, each containing the subset of data.
    """
    data_length = len(data)
    ratios_sum = sum(ratios)
    subsets_length = [int((ratio / ratios_sum) * data_length) for ratio in ratios]
    curr_subset = 0
    while sum(subsets_length) < data_length:
        subsets_length[curr_subset] += 1
        curr_subset += 1
        curr_subset %= len(subsets_length)
    data_indices = list(range(data_length))
    random.shuffle(data_indices)
    subsets = []
    start = 0
    for subset_length in subsets_length:
        end = start + subset_length
        subsets_indices = data_indices[start:end]
        subset = [data[i] for i in subsets_indices]
        subsets.append(subset)
        start = end
    return subsets

