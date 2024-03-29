import torch
from torch.utils.data import Dataset

from torch_utils import stratified_random_split


class TestDataset(Dataset):
    def __init__(self):
        self.length = None
        self.num_classes = None
        self.classes = None
        self.class_counts = None
        self.class_weights = None
        self.feature = None
        self.target = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.feature[idx], self.target[idx]

    def _set_feature(self, feature):
        self.feature = torch.tensor(feature)
        self.length = self.feature.size(0)

    def _set_target(self, target):
        self.target = torch.tensor(target)
        self.length = self.target.size(0)
        self.classes = torch.unique(self.target)
        self.num_classes = self.classes.size(0)
        self.class_counts = torch.tensor([self.target.tolist().count(i) for i in self.classes])
        self.class_weights = 1.0 / self.class_counts

    def _validate(self):
        assert self.feature.size(0) == self.target.size(0)
        assert self.feature.size(0) == self.length
        assert set(self.classes.tolist()) == set(torch.unique(self.target).tolist())
        assert self.num_classes == self.classes.size(0)
        assert sum(self.class_counts) == self.length
        assert torch.all(torch.eq(self.class_weights, 1.0 / self.class_counts))

    def generate(self, shape, num_classes, class_counts=None):
        self.length = shape[0]
        self.num_classes = num_classes
        self.feature = torch.randint(0, 3, shape)
        if class_counts is None:
            self.target = torch.randint(0, self.num_classes, (self.length,))
            self.classes = torch.unique(self.target)
            self.class_counts = torch.tensor([self.target.tolist().count(i) for i in self.classes])
            self.class_weights = 1.0 / self.class_counts
        else:
            assert len(class_counts) == num_classes
            assert sum(class_counts) == self.length
            self.class_counts = torch.tensor(class_counts)
            self.class_weights = 1.0 / self.class_counts
            self.target = torch.cat([torch.full((count,), i) for i, count in enumerate(self.class_counts)])
            randperm = torch.randperm(self.target.size(0))
            self.target = self.target[randperm]
            self.classes = torch.unique(self.target)

    def get_num_classes(self):
        return self.num_classes

    def get_classes(self):
        return self.classes

    def get_class_counts(self):
        return self.class_counts

    def get_class_weights(self):
        return 1.0 / torch.tensor(self.class_counts, dtype=torch.float32)

    def train_test_split(self, train_ratio=0.8):
        train_set = TestDataset()
        test_set = TestDataset()
        train_feature, train_target, test_feature, test_target = stratified_random_split(self.feature, self.target, train_ratio)
        train_set._set_feature(train_feature)
        train_set._set_target(train_target)
        test_set._set_feature(test_feature)
        test_set._set_target(test_target)
        train_set._validate()
        test_set._validate()
        return train_set, test_set


if __name__ == '__main__':
    dataset = TestDataset()
    dataset.generate((37, 10), 3, [9, 17, 11])
    train_set, test_set = dataset.train_test_split()
    print(len(train_set))
    print(len(test_set))
