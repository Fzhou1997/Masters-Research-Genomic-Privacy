from typing import Literal

import torch
from torch import Tensor

from utils_torch.FeatureTargetDataset import FeatureTargetDataset


class ClassificationDataset(FeatureTargetDataset):

    task = Literal['binary', 'multiclass', 'multilabel']

    @property
    def num_classes(self) -> int | list[int]:
        match self.task:
            case 'binary':
                return 2
            case 'multiclass':
                if self.targets.ndim == 1:
                    return torch.unique(self.targets).numel()
                else:
                    return self.targets.shape[1]
            case 'multilabel':


    def _infer_task(self) -> Literal['binary', 'multiclass', 'multilabel']:
        if self.targets.ndim == 1:
            if torch.unique(self.targets).numel() == 2:
                return 'binary'
            else:
                return 'multiclass'
        elif self.targets.ndim > 1:
            if torch.any(torch.sum(self.targets, dim=1) > 1):
                return 'multilabel'
            else:
                return 'multiclass'


