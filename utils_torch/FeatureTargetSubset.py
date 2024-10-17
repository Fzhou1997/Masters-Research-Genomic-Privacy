from typing import Sequence, Sized

from torch import Tensor, Size
from torch.utils.data import Subset

from utils_torch.FeatureTargetDataset import FeatureTargetDataset


class FeatureTargetSubset(Subset[FeatureTargetDataset]):

    dataset: FeatureTargetDataset
    indices: Sequence[int]

    def __init__(self, dataset: FeatureTargetDataset, indices: Sequence[int]):
        super().__init__(dataset, indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        if isinstance(self.indices, list):
            return self.dataset[[self.indices[i] for i in idx]]
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
        if isinstance(self.indices, list):
            return self.dataset.features[[self.indices[i] for i in idx]]
        return self.dataset.features[self.indices[idx]]

    def get_targets(self, idx) -> Tensor:
        if isinstance(self.indices, list):
            return self.dataset.targets[[self.indices[i] for i in idx]]
        return self.dataset.targets[self.indices[idx]]

