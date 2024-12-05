import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, CyclicLR, OneCycleLR
from torchmetrics import Accuracy, Metric

from utils_attacker_lstm.data.DataLoaderAttackerLSTM import DataLoaderAttackerLSTM
from .ModelAttackerLSTMLinear import ModelAttackerLSTMLinear


class TrainerAttackerLSTM:
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
        _num_epochs_trained (int): The number of epochs trained.
        _train_losses (list[float]): List of training losses.
        _train_accuracies (list[float]): List of training accuracies.
        _eval_losses (list[float]): List of evaluation losses.
        _eval_accuracies (list[float]): List of evaluation accuracies.
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

    _num_epochs_trained: int
    _train_losses: list[float]
    _train_accuracies: list[float]
    _eval_losses: list[float]
    _eval_accuracies: list[float]

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
            hx = None
            for snp_batch_index in range(self.train_loader.num_snp_batches):
                data = self.train_loader.get_features_batch(genome_batch_index, snp_batch_index).to(self.device)
                logits, (_, hx) = self.model.forward(data, hx)
            targets = self.train_loader.get_target_batch(genome_batch_index).to(self.device)
            loss = self.criterion(logits, targets)
            loss.backward()
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm, self.norm_type)
            self.optimizer.step()
            if self.scheduler is not None and (
                    isinstance(self.scheduler, CyclicLR) or isinstance(self.scheduler, OneCycleLR)):
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
                hx = None
                for snp_batch_index in range(self.eval_loader.num_snp_batches):
                    data = self.eval_loader.get_features_batch(genome_batch_index, snp_batch_index).to(self.device)
                    logits, (_, hx) = self.model.forward(data, hx)
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
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_loss)
                elif not isinstance(self.scheduler, CyclicLR) and not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
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
        """
        Returns the number of epochs trained.

        Returns:
            int: The number of epochs trained.
        """
        return self._num_epochs_trained

    @property
    def train_losses(self) -> list[float]:
        """
        Returns the list of training losses.

        Returns:
            list[float]: The list of training losses.
        """
        return self._train_losses

    @property
    def train_accuracies(self) -> list[float]:
        """
        Returns the list of training accuracies.

        Returns:
            list[float]: The list of training accuracies.
        """
        return self._train_accuracies

    @property
    def best_train_loss(self) -> float:
        """
        Returns the best training loss.

        Returns:
            float: The best training loss.
        """
        return min(self._train_losses)

    @property
    def best_train_accuracy(self) -> float:
        """
        Returns the best training accuracy.

        Returns:
            float: The best training accuracy.
        """
        return max(self._train_accuracies)

    @property
    def best_train_loss_epoch(self) -> int:
        """
        Returns the epoch with the best training loss.

        Returns:
            int: The epoch with the best training loss.
        """
        return np.argmin(self._train_losses)

    @property
    def best_train_accuracy_epoch(self) -> int:
        """
        Returns the epoch with the best training accuracy.

        Returns:
            int: The epoch with the best training accuracy.
        """
        return np.argmax(self._train_accuracies)

    @property
    def eval_losses(self) -> list[float]:
        """
        Returns the list of evaluation losses.

        Returns:
            list[float]: The list of evaluation losses.
        """
        return self._eval_losses

    @property
    def eval_accuracies(self) -> list[float]:
        """
        Returns the list of evaluation accuracies.

        Returns:
            list[float]: The list of evaluation accuracies.
        """
        return self._eval_accuracies

    @property
    def best_eval_loss(self) -> float:
        """
        Returns the best evaluation loss.

        Returns:
            float: The best evaluation loss.
        """
        return min(self._eval_losses)

    @property
    def best_eval_accuracy(self) -> float:
        """
        Returns the best evaluation accuracy.

        Returns:
            float: The best evaluation accuracy.
        """
        return max(self._eval_accuracies)

    @property
    def best_eval_loss_epoch(self) -> int:
        """
        Returns the epoch with the best evaluation loss.

        Returns:
            int: The epoch with the best evaluation loss.
        """
        return np.argmin(self._eval_losses)

    @property
    def best_eval_accuracy_epoch(self) -> int:
        """
        Returns the epoch with the best evaluation accuracy.

        Returns:
            int: The epoch with the best evaluation accuracy.
        """
        return np.argmax(self._eval_accuracies)

    @property
    def learning_rate(self) -> float:
        """
        Returns the current learning rate.

        Returns:
            float: The current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']
