from utils_torch.FeatureTargetDataset import FeatureTargetDataset


class ClassificationDataset(FeatureTargetDataset):

    @property
    def num_classes(self) -> int:
        return len(self.targets.unique(dim=0))

