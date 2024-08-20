from typing import Tuple

import torch
from torch import Tensor


def stratified_random_split(data: Tensor, labels: Tensor, train_ratio: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    classes = torch.unique(labels)
    train_indices = torch.empty(0, dtype=torch.long)
    test_indices = torch.empty(0, dtype=torch.long)
    for c in classes:
        class_indices = (labels == c).nonzero().squeeze()
        num_class_samples = len(class_indices)
        num_train_samples = round(train_ratio * num_class_samples)
        perm = torch.randperm(num_class_samples)
        train_indices = torch.cat((train_indices, class_indices[perm[:num_train_samples]]))
        test_indices = torch.cat((test_indices, class_indices[perm[num_train_samples:]]))
    train_indices, _ = torch.sort(train_indices)
    test_indices, _ = torch.sort(test_indices)
    train_data = torch.index_select(data, 0, train_indices)
    train_labels = torch.index_select(labels, 0, train_indices)
    test_data = torch.index_select(data, 0, test_indices)
    test_labels = torch.index_select(labels, 0, test_indices)
    return train_data, train_labels, test_data, test_labels
