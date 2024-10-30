import torch
from torch import Tensor

from utils_torch.DatasetFeatureTarget import DatasetFeatureTarget


class BinaryClassificationDataset(DatasetFeatureTarget):

    @property
    def num_samples(self) -> int:
        return len(self)

    @property
    def num_samples_positive(self) -> int:
        return self.targets.sum().item()

    @property
    def num_samples_negative(self) -> int:
        return self.num_samples - self.num_samples_positive

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def classes(self) -> Tensor:
        return torch.tensor([0, 1])

    @property
    def class_counts(self) -> Tensor:
        return torch.tensor([self.num_samples_negative, self.num_samples_positive])

    @property
    def class_weights(self) -> Tensor:
        return 1.0 / self.class_counts

    @property
    def sample_weights(self) -> Tensor:
        return self.class_weights[self.targets.long()]

    @property
    def positive_indices(self) -> Tensor:
        return torch.where(self.targets == 1)[0]

    @property
    def negative_indices(self) -> Tensor:
        return torch.where(self.targets == 0)[0]

    def get_class_indices(self, c: int) -> Tensor:
        return torch.where(self.targets == c)[0]