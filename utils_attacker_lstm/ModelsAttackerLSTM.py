import os.path
from datetime import datetime
from os import PathLike
from typing import Literal

import pandas as pd
import torch.nn as nn

from utils_attacker_lstm import ModelAttackerLSTM

_models_headers = [
    "datetime",
    "target",
    "data_num_snps",
    "model_architecture",
    "model_num_input",
    "model_num_hidden",
    "model_num_layers",
    "model_num_directions",
    "model_dropout",
    "train_num_epochs",
    "train_learning_rate",
    "eval_best_epoch",
    "eval_best_loss",
    "eval_best_acc",
    "test_loss",
    "test_acc",
    "test_precision",
    "test_recall",
    "test_f1_score",
    "test_auroc"
]

class ModelsAttackerLSTM:

    models_path: str | bytes | PathLike[str] | PathLike[bytes]
    models: pd.DataFrame

    def __init__(self,
                 models_path: str | bytes | PathLike[str] | PathLike[bytes],
                 models_file: str = None):
        if models_file is None:
            self.models = pd.DataFrame()
            self.models.columns = _models_headers
        else:
            self.models = pd.read_csv(os.path.join(models_path, models_file))

        pass

    def add_model(self,
                  model: ModelAttackerLSTM,
                  datetime: datetime,
                  target: Literal["pool", "beacon"],
                  data_num_snps: int,
                  model_num_input: int,
                  model_num_hidden: int,
                  model_num_layers: int,
                  model_num_directions: int,
                  model_dropout: float,
                  train_num_epochs: int,
                  train_learning_rate: float,
                  eval_best_epoch: int,
                  eval_best_loss: float,
                  eval_best_acc: float,
                  test_loss: float,
                  test_acc: float,
                  test_precision: float,
                  test_recall: float,
                  test_f1_score: float,
                  test_auroc: float) -> None:
        self.models.loc[len(self.models)] = {
            "datetime": datetime,
            "target": target,
            "data_num_snps": data_num_snps,
            "model_architecture": model.__class__.__name__,
            "model_num_input": model_num_input,
            "model_num_hidden": model_num_hidden,
            "model_num_layers": model_num_layers,
            "model_num_directions": model_num_directions,
            "model_dropout": model_dropout,
            "train_num_epochs": train_num_epochs,
            "train_learning_rate": train_learning_rate,
            "eval_best_epoch": eval_best_epoch,
            "eval_best_loss": eval_best_loss,
            "eval_best_acc": eval_best_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1_score,
            "test_auroc": test_auroc
        }
        if model.__getattr__("save") is None:
            raise ValueError(f"Model {model.__class__.__name__} does not have a save method.")
        # TODO
        pass

    def get_model(self, datetime: datetime) -> nn.Module:
        if datetime not in self.models["datetime"]:
            raise ValueError(f"Model with datetime {datetime} not found.")
        idx = self.models[self.models["datetime"] == datetime].index[0]
        cls = globals()[self.models["model_architecture"][idx]]
        model = cls()
        if model.__getattr__("load") is None:
            raise ValueError(f"Model {model.__class__.__name__} does not have a load method.")
        # TODO
        return model

    pass