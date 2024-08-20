from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.hair_color.dataset import HairColorDataset


class HairColorDataLoader(DataLoader):
    def __init__(self,
                 dataset: HairColorDataset,
                 batch_size=32,
                 shuffle=False,
                 weighted_sampling=False,
                 one_hot_features=False,
                 one_hot_labels=False):
        if weighted_sampling:
            class_weights = dataset.get_label_class_weights()
            sampler = WeightedRandomSampler(dataset.get_label_class_weights(), num_samples=len(dataset), replacement=True)
            super(HairColorDataLoader, self).__init__(dataset, batch_size=batch_size, sampler=sampler)
        else:
            super(HairColorDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.one_hot_features = one_hot_features
        self.one_hot_labels = one_hot_labels

    def __iter__(self):
        for features, labels in super(HairColorDataLoader, self).__iter__():
            features = features.unsqueeze(2)
            if self.one_hot_features:
                features = one_hot(features, num_classes=3)
            if self.one_hot_labels:
                labels = one_hot(labels.long(), num_classes=3)
            yield features.float(), labels.float()

