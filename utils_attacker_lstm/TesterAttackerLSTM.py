import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, ConfusionMatrix

from utils_attacker_lstm import ModelAttackerLSTMLinear, DataLoaderAttackerLSTM


class LSTMAttackerTester:
    """
    LSTMAttackerTester is responsible for testing the LSTMAttacker model using various metrics.

    Attributes:
        model (ModelAttackerLSTMLinear): The LSTM model to be tested.
        criterion (nn.Module): The loss function.
        test_loader (DataLoaderAttackerLSTM): DataLoader for testing data.
        device (torch.device): The device to run the model on (CPU or GPU).
        accuracy (Accuracy): Metric for calculating accuracy.
        f1 (F1Score): Metric for calculating F1 score.
        precision (Precision): Metric for calculating precision.
        recall (Recall): Metric for calculating recall.
        auroc (AUROC): Metric for calculating AUROC.
        confusion_matrix (ConfusionMatrix): Metric for calculating the confusion matrix.
    """

    _loss: float
    _accuracy_score: float
    _precision_score: float
    _recall_score: float
    _f1_score: float
    _auroc_score: float
    _confusion_matrix_scores: list[list[int]]

    def __init__(self,
                 model: ModelAttackerLSTMLinear,
                 criterion: nn.Module,
                 test_loader: DataLoaderAttackerLSTM,
                 device: torch.device):
        """
        Initializes the LSTMAttackerTester.

        Args:
            model (ModelAttackerLSTMLinear): The LSTM model to be tested.
            criterion (nn.Module): The loss function.
            test_loader (DataLoaderAttackerLSTM): DataLoader for testing data.
            device (torch.device): The device to run the model on (CPU or GPU).
        """
        self.model = model
        self.criterion = criterion
        self.test_loader = test_loader
        self.device = device
        self.accuracy = Accuracy(task='binary').to(device)
        self.f1 = F1Score(task='binary').to(device)
        self.precision = Precision(task='binary').to(device)
        self.recall = Recall(task='binary').to(device)
        self.auroc = AUROC(task='binary').to(device)
        self.confusion_matrix = ConfusionMatrix(task='binary').to(device)

        self._loss = 0
        self._accuracy_score = 0
        self._precision_score = 0
        self._recall_score = 0
        self._f1_score = 0
        self._auroc_score = 0
        self._confusion_matrix_scores = []

    def test(self):
        """
        Tests the model and evaluates its performance using various metrics.

        Returns:
            tuple: A tuple containing the average loss, accuracy, precision, recall, F1 score, AUROC, and confusion matrix.
        """
        self.model.eval()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()
        self.confusion_matrix.reset()

        self._loss = 0
        self._accuracy_score = 0
        self._precision_score = 0
        self._recall_score = 0
        self._f1_score = 0
        self._auroc_score = 0
        self._confusion_matrix_scores = []

        with torch.no_grad():
            for genome_batch_index in range(self.test_loader.num_genome_batches):
                hidden, cell = self.model.init_hidden_cell(self.test_loader.get_genome_batch_size(genome_batch_index))
                hidden, cell = hidden.to(self.device), cell.to(self.device)
                for snp_batch_index in range(self.test_loader.num_snp_batches):
                    features = self.test_loader.get_features_batch(genome_batch_index, snp_batch_index).to(self.device)
                    (hidden, cell), logits = self.model(features, hidden, cell)
                targets = self.test_loader.get_target_batch(genome_batch_index).to(self.device)
                self._loss += self.criterion(logits, targets).item()
                pred = self.model.classify(self.model.predict(logits)).long()
                true = targets.long()
                self.accuracy.update(pred, true)
                self.precision.update(pred, true)
                self.recall.update(pred, true)
                self.f1.update(pred, true)
                self.auroc.update(pred, true)
                self.confusion_matrix.update(pred, true)
        self._loss /= self.test_loader.num_genome_batches
        self._accuracy_score = self.accuracy.compute().cpu().item()
        self._precision_score = self.precision.compute().cpu().item()
        self._recall_score = self.recall.compute().cpu().item()
        self._f1_score = self.f1.compute().cpu().item()
        self._auroc_score = self.auroc.compute().cpu().item()
        self._confusion_matrix_scores = self.confusion_matrix.compute().cpu().tolist()

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def accuracy_score(self) -> float:
        return self._accuracy_score

    @property
    def precision_score(self) -> float:
        return self._precision_score

    @property
    def recall_score(self) -> float:
        return self._recall_score

    @property
    def f1_score(self) -> float:
        return self._f1_score

    @property
    def auroc_score(self) -> float:
        return self._auroc_score

    @property
    def confusion_matrix_scores(self) -> list[list[int]]:
        return self._confusion_matrix_scores
