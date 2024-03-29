from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, WeightedRandomSampler


class TestDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, weighted_sampling=False, one_hot_features=False, one_hot_targets=False):
        if weighted_sampling:
            sampler = WeightedRandomSampler(weights=dataset.get_class_weights(),
                                            num_samples=len(dataset),
                                            replacement=True)
        else:
            sampler = None
        super(TestDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
        self.typed_dataset = dataset
        self.one_hot_features = one_hot_features
        self.one_hot_targets = one_hot_targets

    def __iter__(self):
        for feature, target in super(TestDataLoader, self).__iter__():

            if self.one_hot_targets:
                target = one_hot(target, num_classes=self.typed_dataset.get_num_classes())
            yield feature, target

