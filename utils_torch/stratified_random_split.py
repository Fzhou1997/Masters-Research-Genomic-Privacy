import random
from typing import Any

import torch

from .FeatureTargetDataset import FeatureTargetDataset
from .FeatureTargetSubset import FeatureTargetSubset


def stratified_random_split(
        dataset: FeatureTargetDataset,
        ratios: list[float]) -> list[FeatureTargetSubset]:
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
    subsets = [FeatureTargetSubset(dataset, subset_indices) for subset_indices in subsets_indices]
    return subsets

def _random_split(
        data: list[Any],
        ratios: list[float]) -> list[list[Any]]:
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

