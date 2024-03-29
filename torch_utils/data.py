from typing import Tuple

import torch
from torch import Tensor


def stratified_random_split(features: Tensor, targets: Tensor, train_ratio: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    classes = torch.unique(targets)
    train_indices = []
    test_indices = []
    for c in classes:
        class_indices = (targets == c).nonzero().squeeze()
        num_class_samples = len(class_indices)
        num_train_samples = round(train_ratio * num_class_samples)
        perm = torch.randperm(num_class_samples)
        train_indices.extend(class_indices[perm[:num_train_samples]])
        test_indices.extend(class_indices[perm[num_train_samples:]])
    train_indices = sorted(train_indices)
    test_indices = sorted(test_indices)
    train_features = torch.index_select(features, 0, torch.tensor(train_indices))
    train_targets = torch.index_select(targets, 0, torch.tensor(train_indices))
    test_features = torch.index_select(features, 0, torch.tensor(test_indices))
    test_targets = torch.index_select(targets, 0, torch.tensor(test_indices))
    return train_features, train_targets, test_features, test_targets
