from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.hair_color.hair_color_dataset import HairColorDataset


class HairColorDataLoader(DataLoader):
    def __init__(self,
                 dataset: HairColorDataset,
                 batch_size=32,
                 shuffle=False,
                 weighted_sampling=False,
                 one_hot_data=False,
                 one_hot_labels=False):
        sampler = None
        if weighted_sampling:
            sampler = WeightedRandomSampler(weights=dataset.get_label_class_weights(), num_samples=len(dataset), replacement=True)
        super(HairColorDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
        self.hair_color_dataset = dataset
        self.one_hot_data = one_hot_data
        self.one_hot_labels = one_hot_labels

    def __iter__(self):
        for data, labels in super(HairColorDataLoader, self).__iter__():
            data = data.unsqueeze(2)
            if self.one_hot_data:
                data = one_hot(data, num_classes=self.hair_color_dataset.get_num_data_classes())
            if self.one_hot_labels:
                labels = one_hot(labels, num_classes=self.hair_color_dataset.get_num_label_classes())
            yield data, labels

    def is_data_one_hot(self):
        return self.one_hot_data

    def is_labels_one_hot(self):
        return self.one_hot_labels


