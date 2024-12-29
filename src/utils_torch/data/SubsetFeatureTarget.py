from typing import Sequence

from torch import Tensor, Size
from torch.utils.data import Subset

from .DatasetFeatureTarget import DatasetFeatureTarget


class SubsetFeatureTarget(Subset[DatasetFeatureTarget]):

    dataset: DatasetFeatureTarget
    indices: Sequence[int]

    def __init__(self, dataset: DatasetFeatureTarget, indices: Sequence[int]):
        super().__init__(dataset, indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        if isinstance(idx, tuple) and isinstance(idx[0], list):
            indices = list(idx)
            indices[0] = [self.indices[i] for i in idx[0]]
            return self.dataset[tuple(indices)]
        return self.dataset[self.indices[idx]]

    @property
    def shape(self) -> Size:
        dataset_size = list(self.dataset.shape)
        dataset_size[0] = len(self)
        return Size(tuple(dataset_size))

    @property
    def features(self) -> Tensor:
        return self.dataset.features[self.indices]

    @property
    def targets(self) -> Tensor:
        return self.dataset.targets[self.indices]

    def get_features(self, idx) -> Tensor:
        if isinstance(idx, list):
            return self.dataset.features[[self.indices[i] for i in idx]]
        if isinstance(idx, tuple) and isinstance(idx[0], list):
            indices = list(idx)
            indices[0] = [self.indices[i] for i in idx[0]]
            return self.dataset.features[tuple(indices)]
        return self.dataset.features[self.indices[idx]]

    def get_targets(self, idx) -> Tensor:
        if isinstance(idx, list):
            return self.dataset.targets[[self.indices[i] for i in idx]]
        return self.dataset.targets[self.indices[idx]]

