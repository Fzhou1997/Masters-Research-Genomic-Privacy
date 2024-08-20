import os
from typing import Self

import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils_genomes import *
from utils_torch import stratified_random_split

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class HairColorDataset(Dataset):
    def __init__(self):
        self.features = None
        self.feature_length = None
        self.feature_classes = None
        self.num_feature_classes = None
        self.feature_class_counts = None
        self.feature_class_weights = None
        self.labels = None
        self.label_length = None
        self.label_classes = None
        self.num_label_classes = None
        self.label_class_counts = None
        self.label_class_weights = None

    def __len__(self):
        return self.feature_length

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def _set_features(self, features: Tensor) -> None:
        self.features = features
        self.feature_length = self.features.size(0)
        self.feature_classes, self.feature_class_counts = torch.unique(self.features, return_counts=True)
        self.num_feature_classes = self.feature_classes.size(0)
        self.feature_class_weights = 1.0 / self.feature_class_counts

    def _set_labels(self, labels: Tensor) -> None:
        self.labels = labels
        self.label_length = self.labels.size(0)
        self.label_classes, self.label_class_counts = torch.unique(self.labels, return_counts=True)
        self.num_label_classes = self.label_classes.size(0)
        self.label_class_weights = 1.0 / self.label_class_counts

    def get_data_classes(self) -> Tensor:
        return self.feature_classes

    def get_num_data_classes(self) -> int:
        return self.num_feature_classes

    def get_data_class_counts(self) -> Tensor:
        return self.feature_class_counts

    def get_data_class_weights(self) -> Tensor:
        return self.feature_class_weights

    def get_label_classes(self) -> Tensor:
        return self.label_classes

    def get_num_label_classes(self) -> int:
        return self.num_label_classes

    def get_label_class_counts(self) -> Tensor:
        return self.label_class_counts

    def get_label_class_weights(self) -> Tensor:
        return self.label_class_weights

    def split_train_test(self, train_ratio=0.8) -> tuple[Self, Self]:
        train_set = HairColorDataset()
        test_set = HairColorDataset()
        train_data, train_labels, test_data, test_labels = stratified_random_split(self.features, self.labels, train_ratio)
        train_set._set_features(train_data)
        train_set._set_labels(train_labels)
        test_set._set_features(test_data)
        test_set._set_labels(test_labels)
        return train_set, test_set

    def from_genomes(self, genomes: Genomes) -> None:
        self._set_features(genomes.get_genotypes_tensor().int())
        self._set_labels(genomes.get_phenotypes_tensor().int())

    def load(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes], prefix: str) -> Self:
        self._set_features(torch.load(f'{path}/{prefix}_genotypes.pt'))
        self._set_labels(torch.load(f'{path}/{prefix}_phenotypes.pt'))
        return self

    def save(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes], prefix: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.features, f'{path}/{prefix}_genotypes.pt')
        torch.save(self.labels, f'{path}/{prefix}_phenotypes.pt')
