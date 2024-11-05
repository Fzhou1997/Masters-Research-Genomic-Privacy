import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Accuracy, Metric

from . import DataLoaderAttackerLSTM
from .ModelAttackerLSTMLinear import ModelAttackerLSTMLinear


class LSTMAttackerTrainer:
    """
    LSTMAttackerTrainer is responsible for training the LSTMAttacker model.

    Attributes:
        model (ModelAttackerLSTMLinear): The LSTM model to be trained.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer for training.
        scheduler (LRScheduler): The learning rate scheduler.
        max_grad_norm (float): The maximum gradient norm for clipping.
        norm_type (float): The type of norm for clipping.
        train_loader (DataLoaderAttackerLSTM): DataLoader for training data.
        eval_loader (DataLoaderAttackerLSTM): DataLoader for evaluation data.
        device (torch.device): The device to run the model on (CPU or GPU).
        accuracy (Metric): The accuracy metric.
    """

    model: ModelAttackerLSTMLinear
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: LRScheduler
    max_grad_norm: float
    norm_type: float
    train_loader: DataLoaderAttackerLSTM
    eval_loader: DataLoaderAttackerLSTM
    device: torch.device
    accuracy: Metric

    def __init__(self,
                 model: ModelAttackerLSTMLinear,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: DataLoaderAttackerLSTM,
                 eval_loader: DataLoaderAttackerLSTM,
                 device: torch.device,
                 scheduler: LRScheduler = None,
                 max_grad_norm: float = None,
                 norm_type: float = 2):
        """
        Initializes the LSTMAttackerTrainer.

        Args:
            model (ModelAttackerLSTMLinear): The LSTM model to be trained.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer for training.
            train_loader (DataLoaderAttackerLSTM): DataLoader for training data.
            eval_loader (DataLoaderAttackerLSTM): DataLoader for evaluation data.
            device (torch.device): The device to run the model on (CPU or GPU).
            scheduler (LRScheduler, optional): The learning rate scheduler. Default is None.
            max_grad_norm (float, optional): The maximum gradient norm for clipping. Default is None.
            norm_type (float, optional): The type of norm for clipping. Default is 2.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.norm_type = norm_type
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.accuracy = Accuracy(task='binary').to(device)

        self._num_epochs_trained = 0
        self._train_losses = []
        self._train_accuracies = []
        self._eval_losses = []
        self._eval_accuracies = []

    def _train_one_epoch(self) -> None:
        """
        Trains the model for one epoch.

        Returns:
            tuple: A tuple containing the average loss and accuracy for the epoch.
        """
        running_loss = 0
        self.accuracy.reset()
        self.model.train()
        for genome_batch_index in range(self.train_loader.num_genome_batches):
            self.optimizer.zero_grad()
            hidden, cell = self.model.get_hx(self.train_loader.get_genome_batch_size(genome_batch_index))
            hidden, cell = hidden.to(self.device), cell.to(self.device)
            for snp_batch_index in range(self.train_loader.num_snp_batches):
                data = self.train_loader.get_features_batch(genome_batch_index, snp_batch_index).to(self.device)
                (hidden, cell), logits = self.model(data, hidden, cell)
            targets = self.train_loader.get_target_batch(genome_batch_index).to(self.device)
            loss = self.criterion(logits, targets)
            loss.backward()
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm, self.norm_type)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            running_loss += loss.item()
            pred = self.model.classify(self.model.predict(logits)).long()
            true = targets.long()
            self.accuracy.update(pred, true)
        running_loss /= self.train_loader.num_genome_batches
        accuracy = self.accuracy.compute().cpu().item()
        self._num_epochs_trained += 1
        self._train_losses.append(running_loss)
        self._train_accuracies.append(accuracy)

    def _eval_one_epoch(self) -> None:
        """
        Evaluates the model for one epoch.

        Returns:
            tuple: A tuple containing the average loss and accuracy for the epoch.
        """
        loss = 0
        self.accuracy.reset()
        self.model.eval()
        with torch.no_grad():
            for genome_batch_index in range(self.eval_loader.num_genome_batches):
                hidden, cell = self.model.get_hx(self.eval_loader.get_genome_batch_size(genome_batch_index))
                hidden, cell = hidden.to(self.device), cell.to(self.device)
                for snp_batch_index in range(self.eval_loader.num_snp_batches):
                    data = self.eval_loader.get_features_batch(genome_batch_index, snp_batch_index).to(self.device)
                    (hidden, cell), logits = self.model(data, hidden, cell)
                targets = self.eval_loader.get_target_batch(genome_batch_index).to(self.device)
                loss += self.criterion(logits, targets).item()
                pred = self.model.classify(self.model.predict(logits)).long()
                true = targets.long()
                self.accuracy.update(pred, true)
        loss /= self.eval_loader.num_genome_batches
        accuracy = self.accuracy.compute().cpu().item()
        self._eval_losses.append(loss)
        self._eval_accuracies.append(accuracy)

    def train(self,
              num_epochs: int,
              verbose: bool = False) -> None:
        """
        Trains the model for a specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model.
            verbose (bool, optional): If True, prints training progress. Default is False.

        Returns:
            tuple: A tuple containing lists of training losses, training accuracies, evaluation losses, and evaluation accuracies.
        """
        best_epoch = -1
        best_loss = float('inf')
        best_state_dict = None
        for epoch in range(num_epochs):
            self._train_one_epoch()
            self._eval_one_epoch()
            train_loss = self._train_losses[-1]
            train_accuracy = self._train_accuracies[-1]
            eval_loss = self._eval_losses[-1]
            eval_accuracy = self._eval_accuracies[-1]
            if verbose:
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}')
                print(f'Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {eval_accuracy:.2f}')
            if eval_loss < best_loss:
                if verbose:
                    print(f'Evaluation Loss Decreased: {best_loss:.4f} -> {eval_loss:.4f}. Saving Model...')
                best_epoch = epoch
                best_loss = eval_loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
            if verbose:
                print(25 * "==")
        if verbose:
            print(f'Best Model Found at Epoch {best_epoch + 1}.')
        self.model.load_state_dict(best_state_dict)

    @property
    def num_epoches_trained(self) -> int:
        return self._num_epochs_trained

    @property
    def train_losses(self) -> list[float]:
        return self._train_losses

    @property
    def train_accuracies(self) -> list[float]:
        return self._train_accuracies

    @property
    def best_train_loss(self) -> float:
        return min(self._train_losses)

    @property
    def best_train_accuracy(self) -> float:
        return max(self._train_accuracies)

    @property
    def best_train_loss_epoch(self) -> int:
        return np.argmin(self._train_losses)

    @property
    def best_train_accuracy_epoch(self) -> int:
        return np.argmax(self._train_accuracies)

    @property
    def eval_losses(self) -> list[float]:
        return self._eval_losses

    @property
    def eval_accuracies(self) -> list[float]:
        return self._eval_accuracies

    @property
    def best_eval_loss(self) -> float:
        return min(self._eval_losses)

    @property
    def best_eval_accuracy(self) -> float:
        return max(self._eval_accuracies)

    @property
    def best_eval_loss_epoch(self) -> int:
        return np.argmin(self._eval_losses)

    @property
    def best_eval_accuracy_epoch(self) -> int:
        return np.argmax(self._eval_accuracies)

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]['lr']