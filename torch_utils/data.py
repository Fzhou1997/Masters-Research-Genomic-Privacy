from typing import Tuple

import torch
from torch import Tensor


def stratified_random_split(data: Tensor, labels: Tensor, train_ratio: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    classes = torch.unique(labels)
    train_indices = []
    test_indices = []
    for c in classes:
        class_indices = (labels == c).nonzero().squeeze()
        num_class_samples = len(class_indices)
        num_train_samples = round(train_ratio * num_class_samples)
        perm = torch.randperm(num_class_samples)
        train_indices.extend(class_indices[perm[:num_train_samples]])
        test_indices.extend(class_indices[perm[num_train_samples:]])
    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)
    train_data = torch.index_select(data, 0, torch.tensor(train_indices))
    train_labels = torch.index_select(labels, 0, torch.tensor(train_indices))
    test_data = torch.index_select(data, 0, torch.tensor(test_indices))
    test_labels = torch.index_select(labels, 0, torch.tensor(test_indices))
    return train_data, train_labels, test_data, test_labels
