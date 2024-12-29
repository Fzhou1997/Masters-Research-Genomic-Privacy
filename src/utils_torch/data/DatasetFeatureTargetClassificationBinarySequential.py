from .DatasetFeatureTargetClassificationBinary import DatasetFeatureTargetClassificationBinary


class DatasetFeatureTargetClassificationBinarySequential(DatasetFeatureTargetClassificationBinary):

    @property
    def num_timesteps(self) -> int:
        return self.features.shape[1]