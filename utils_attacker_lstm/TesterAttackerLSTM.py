import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, ConfusionMatrix

from utils_attacker_lstm import ModelAttackerLSTM, LSTMAttackerDataLoader


class LSTMAttackerTester:
    """
    LSTMAttackerTester is responsible for testing the LSTMAttacker model using various metrics.

    Attributes:
        model (ModelAttackerLSTM): The LSTM model to be tested.
        criterion (nn.Module): The loss function.
        test_loader (LSTMAttackerDataLoader): DataLoader for testing data.
        device (torch.device): The device to run the model on (CPU or GPU).
        accuracy (Accuracy): Metric for calculating accuracy.
        f1_score (F1Score): Metric for calculating F1 score.
        precision (Precision): Metric for calculating precision.
        recall (Recall): Metric for calculating recall.
        auroc (AUROC): Metric for calculating AUROC.
        confusion_matrix (ConfusionMatrix): Metric for calculating the confusion matrix.
    """

    def __init__(self,
                 model: ModelAttackerLSTM,
                 criterion: nn.Module,
                 test_loader: LSTMAttackerDataLoader,
                 device: torch.device):
        """
        Initializes the LSTMAttackerTester.

        Args:
            model (ModelAttackerLSTM): The LSTM model to be tested.
            criterion (nn.Module): The loss function.
            test_loader (LSTMAttackerDataLoader): DataLoader for testing data.
            device (torch.device): The device to run the model on (CPU or GPU).
        """
        self.model = model
        self.criterion = criterion
        self.test_loader = test_loader
        self.device = device
        self.accuracy = Accuracy(task='binary').to(device)
        self.f1_score = F1Score(task='binary').to(device)
        self.precision = Precision(task='binary').to(device)
        self.recall = Recall(task='binary').to(device)
        self.auroc = AUROC(task='binary').to(device)
        self.confusion_matrix = ConfusionMatrix(task='binary').to(device)

    def test(self) -> tuple[float, float, float, float, float, float, list[list[int]]]:
        """
        Tests the model and evaluates its performance using various metrics.

        Returns:
            tuple: A tuple containing the average loss, accuracy, precision, recall, F1 score, AUROC, and confusion matrix.
        """
        self.model.eval()
        loss = 0
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        self.auroc.reset()
        self.confusion_matrix.reset()
        with torch.no_grad():
            for genome_batch_index in range(self.test_loader.num_genome_batches):
                hidden, cell = self.model.init_hidden_cell(self.test_loader.get_genome_batch_size(genome_batch_index))
                hidden, cell = hidden.to(self.device), cell.to(self.device)
                for snp_batch_index in range(self.test_loader.num_snp_batches):
                    features = self.test_loader.get_features_batch(genome_batch_index, snp_batch_index).to(self.device)
                    (hidden, cell), logits = self.model(features, hidden, cell)
                targets = self.test_loader.get_target_batch(genome_batch_index).to(self.device)
                loss += self.criterion(logits, targets).item()
                pred = self.model.classify(self.model.predict(logits)).long()
                true = targets.long()
                self.accuracy.update(pred, true)
                self.precision.update(pred, true)
                self.recall.update(pred, true)
                self.f1_score.update(pred, true)
                self.auroc.update(pred, true)
                self.confusion_matrix.update(pred, true)
        loss /= self.test_loader.num_genome_batches
        accuracy = self.accuracy.compute().cpu().item()
        precision = self.precision.compute().cpu().item()
        recall = self.recall.compute().cpu().item()
        f1_score = self.f1_score.compute().cpu().item()
        auroc = self.auroc.compute().cpu().item()
        cm = self.confusion_matrix.compute().cpu().tolist()
        return loss, accuracy, precision, recall, f1_score, auroc, cm
