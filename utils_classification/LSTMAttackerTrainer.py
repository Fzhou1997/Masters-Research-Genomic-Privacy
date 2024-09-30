import torch
import torch.nn as nn
import torch.optim as optim

from .LSTMAttacker import LSTMAttacker


class LSTMAttackerTrainer:
    def __init__(self,
                 model: LSTMAttacker,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device