from utils_torch.BinaryClassificationDataset import BinaryClassificationDataset


class SequentialBinaryClassificationDataset(BinaryClassificationDataset):

    @property
    def num_timesteps(self) -> int:
        return self.features.shape[1]