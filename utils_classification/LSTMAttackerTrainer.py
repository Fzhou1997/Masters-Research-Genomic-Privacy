import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Accuracy

from . import LSTMAttackerDataLoader
from .LSTMAttacker import LSTMAttacker


class LSTMAttackerTrainer:
    def __init__(self,
                 model: LSTMAttacker,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: LRScheduler,
                 train_loader: LSTMAttackerDataLoader,
                 test_loader: LSTMAttackerDataLoader,
                 device: torch.device):
        self.model = model
        self.criterion = criterion
        self.accuracy = Accuracy(task='binary').to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def _train_one_epoch(self) -> tuple[float, float]:
        running_loss = 0
        self.accuracy.reset()
        self.model.train()
        for genome_batch_index in range(self.train_loader.num_genome_batches):
            self.optimizer.zero_grad()
            for snp_batch_index in range(self.train_loader.num_snp_batches):
                data = self.train_loader.get_data_batch(genome_batch_index, snp_batch_index)

        return total_loss / len(train_loader)

    def _eval_one_epoch(self):

    def train(self, ) -> float:
