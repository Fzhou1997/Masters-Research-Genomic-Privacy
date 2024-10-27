import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Accuracy

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
        train_loader (DataLoaderAttackerLSTM): DataLoader for training data.
        eval_loader (DataLoaderAttackerLSTM): DataLoader for evaluation data.
        device (torch.device): The device to run the model on (CPU or GPU).
    """

    def __init__(self,
                 model: ModelAttackerLSTMLinear,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: DataLoaderAttackerLSTM,
                 eval_loader: DataLoaderAttackerLSTM,
                 device: torch.device,
                 scheduler: LRScheduler = None):
        """
        Initializes the LSTMAttackerTrainer.

        Args:
            model (ModelAttackerLSTMLinear): The LSTM model to be trained.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer for training.
            scheduler (LRScheduler): The learning rate scheduler.
            train_loader (DataLoaderAttackerLSTM): DataLoader for training data.
            eval_loader (DataLoaderAttackerLSTM): DataLoader for evaluation data.
            device (torch.device): The device to run the model on (CPU or GPU).
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.accuracy = Accuracy(task='binary').to(device)

    def _train_one_epoch(self) -> tuple[float, float]:
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
            hidden, cell = self.model.init_hidden_cell(self.train_loader.get_genome_batch_size(genome_batch_index))
            hidden, cell = hidden.to(self.device), cell.to(self.device)
            for snp_batch_index in range(self.train_loader.num_snp_batches):
                data = self.train_loader.get_features_batch(genome_batch_index, snp_batch_index).to(self.device)
                (hidden, cell), logits = self.model(data, hidden, cell)
            targets = self.train_loader.get_target_batch(genome_batch_index).to(self.device)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            running_loss += loss.item()
            pred = self.model.classify(self.model.predict(logits)).long()
            true = targets.long()
            self.accuracy.update(pred, true)
        running_loss /= self.train_loader.num_genome_batches
        accuracy = self.accuracy.compute().cpu().item()
        return running_loss, accuracy

    def _eval_one_epoch(self) -> tuple[float, float]:
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
                hidden, cell = self.model.init_hidden_cell(self.eval_loader.get_genome_batch_size(genome_batch_index))
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
        return loss, accuracy

    def train(self,
              num_epochs: int,
              verbose: bool = False) -> tuple[list[float], list[float], list[float], list[float]]:
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
        train_losses = []
        train_accuracies = []
        eval_losses = []
        eval_accuracies = []
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self._train_one_epoch()
            eval_loss, eval_accuracy = self._eval_one_epoch()
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            eval_losses.append(eval_loss)
            eval_accuracies.append(eval_accuracy)
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
        return train_losses, train_accuracies, eval_losses, eval_accuracies
