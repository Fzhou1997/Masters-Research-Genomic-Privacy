from torch import Tensor
from torch.utils.data import Dataset


class FeatureTargetDataset(Dataset[tuple[Tensor, Tensor]]):

    features: Tensor
    targets: Tensor

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.targets[idx]

    def get_features(self, idx) -> Tensor:
        return self.features[idx]

    def get_targets(self, idx) -> Tensor:
        return self.targets[idx]