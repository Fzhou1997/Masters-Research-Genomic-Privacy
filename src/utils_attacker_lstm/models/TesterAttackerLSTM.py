import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, ROC, ConfusionMatrix

from src.utils_attacker_lstm.data.DataLoaderAttackerLSTM import DataLoaderAttackerLSTM
from .ModelAttackerLSTM import ModelAttackerLSTM


class TesterAttackerLSTM:
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
    _roc_curve: tuple[list[float], list[float], list[float]]
    _confusion_matrix_scores: list[list[int]]

    def __init__(self,
                 model: ModelAttackerLSTM,
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
        self.roc = ROC(task='binary').to(device)
        self.confusion_matrix = ConfusionMatrix(task='binary').to(device)

        self._loss = 0
        self._accuracy_score = 0
        self._precision_score = 0
        self._recall_score = 0
        self._f1_score = 0
        self._auroc_score = 0
        self._roc_curve = ([], [], [])
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
        self.roc.reset()
        self.confusion_matrix.reset()

        self._loss = 0
        self._accuracy_score = 0
        self._precision_score = 0
        self._recall_score = 0
        self._f1_score = 0
        self._auroc_score = 0
        self._roc_curve = ([], [], [])
        self._confusion_matrix_scores = []

        with torch.no_grad():
            for genome_batch_index in range(self.test_loader.num_genome_batches):
                hx = None
                for snp_batch_index in range(self.test_loader.num_snp_batches):
                    data = self.test_loader.get_features_batch(genome_batch_index, snp_batch_index).to(self.device)
                    logits, out = self.model.forward(data, hx)
                    hx = []
                    for i, out_i in enumerate(out):
                        _, hx_i = out_i
                        hx.append(hx_i)
                    hx = tuple(hx)
                targets = self.test_loader.get_target_batch(genome_batch_index).to(self.device)
                self._loss += self.criterion(logits, targets).item()
                pred = self.model.predict(logits)
                clas = self.model.classify(pred).long()
                true = targets.long()
                self.accuracy.update(clas, true)
                self.precision.update(clas, true)
                self.recall.update(clas, true)
                self.f1.update(clas, true)
                self.auroc.update(clas, true)
                self.roc.update(pred, true)
                self.confusion_matrix.update(clas, true)
        self._loss /= self.test_loader.num_genome_batches
        self._accuracy_score = self.accuracy.compute().cpu().item()
        self._precision_score = self.precision.compute().cpu().item()
        self._recall_score = self.recall.compute().cpu().item()
        self._f1_score = self.f1.compute().cpu().item()
        self._auroc_score = self.auroc.compute().cpu().item()
        fpr, tpr, thresholds = self.roc.compute()
        self._roc_curve = (fpr.cpu().tolist(), tpr.cpu().tolist(), thresholds.cpu().tolist())
        self._confusion_matrix_scores = self.confusion_matrix.compute().cpu().tolist()

    @property
    def loss(self) -> float:
        """
        Returns the average loss.

        Returns:
            float: The average loss.
        """
        return self._loss

    @property
    def accuracy_score(self) -> float:
        """
        Returns the accuracy score.

        Returns:
            float: The accuracy score.
        """
        return self._accuracy_score

    @property
    def precision_score(self) -> float:
        """
        Returns the precision score.

        Returns:
            float: The precision score.
        """
        return self._precision_score

    @property
    def recall_score(self) -> float:
        """
        Returns the recall score.

        Returns:
            float: The recall score.
        """
        return self._recall_score

    @property
    def f1_score(self) -> float:
        """
        Returns the F1 score.

        Returns:
            float: The F1 score.
        """
        return self._f1_score

    @property
    def auroc_score(self) -> float:
        """
        Returns the AUROC score.

        Returns:
            float: The AUROC score.
        """
        return self._auroc_score

    @property
    def roc_curve(self) -> tuple[list[float], list[float], list[float]]:
        """
        Returns the ROC curve.

        Returns:
            tuple[list[float], list[float], list[float]]: The ROC curve.
        """
        return self._roc_curve

    @property
    def confusion_matrix_scores(self) -> list[list[int]]:
        """
        Returns the confusion matrix scores.

        Returns:
            list[list[int]]: The confusion matrix scores.
        """
        return self._confusion_matrix_scores
