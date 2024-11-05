import os.path
from os import PathLike

import pandas as pd

from utils_attacker_lstm import ModelAttackerLSTM, DatasetAttackerLSTM, DataLoaderAttackerLSTM, TrainerAttackerLSTM, \
    TesterAttackerLSTM, ModelAttackerLSTMLinear, ModelAttackerConvLSTMLinear

# Headers for the models DataFrame
_models_headers = [
    "id",
    "data_structure",
    "data_num_snps",
    "loader_genome_batch_size",
    "loader_snp_batch_size",
    "model_architecture",
    "model_conv_in_channels",
    "model_conv_out_channels",
    "model_conv_kernel_size",
    "model_conv_stride",
    "model_lstm_input_size",
    "model_lstm_hidden_size",
    "model_lstm_num_layers",
    "model_lstm_bidirectional",
    "model_lstm_dropout",
    "model_linear_out_features",
    "trainer_num_epochs",
    "trainer_learning_rate",
    "trainer_best_eval_epoch",
    "trainer_best_eval_loss",
    "trainer_best_eval_acc",
    "tester_loss",
    "tester_acc",
    "tester_precision",
    "tester_recall",
    "tester_f1_score",
    "tester_auroc"
]

class ManagerAttackerLSTM:
    """
    Manager class for handling different LSTM-based attacker models.
    """

    models_dir: str | bytes | PathLike[str] | PathLike[bytes]
    models: pd.DataFrame

    def __init__(self,
                 models_dir: str | bytes | PathLike[str] | PathLike[bytes],
                 models_file: str = None):
        """
        Initialize the ManagerAttackerLSTM.

        :param models_dir: Directory where models are stored.
        :param models_file: Optional CSV file containing model information.
        """
        if models_file is None:
            self.models = pd.DataFrame(columns=_models_headers)
        else:
            self.models = pd.read_csv(os.path.join(models_dir, models_file))
            for header in _models_headers:
                if header not in self.models.columns:
                    self.models[header] = pd.NA
            self.models = self.models[_models_headers]
        self.models.set_index("id", inplace=True)
        self.models_dir = models_dir

    def add_model(self,
                  model_id: str,
                  data: DatasetAttackerLSTM,
                  loader: DataLoaderAttackerLSTM,
                  model: ModelAttackerLSTM,
                  trainer: TrainerAttackerLSTM,
                  tester: TesterAttackerLSTM) -> None:
        """
        Add a model to the DataFrame.

        :param model_id: Unique identifier for the model.
        :param data: Dataset used by the model.
        :param loader: DataLoader used by the model.
        :param model: Model instance.
        :param trainer: Trainer instance.
        :param tester: Tester instance.
        """
        model_architecture = model.__class__.__name__
        match model_architecture:
            case "ModelAttackerLSTMLinear":
                self._add_lstm_linear_model(model_id, data, loader, model, trainer, tester)
            case "ModelAttackerConvLSTMLinear":
                self._add_conv_lstm_linear_model(model_id, data, loader, model, trainer, tester)
            case _:
                raise ValueError(f"Model architecture {model_architecture} not supported.")

    def get_model(self, model_id: str) -> ModelAttackerLSTM:
        """
        Retrieve a model based on its ID.

        :param model_id: Unique identifier for the model.
        :return: Model instance.
        """
        if model_id not in self.models["datetime"]:
            raise ValueError(f"Model with id {model_id} not found.")
        model_architecture = self.models.loc[model_id, "model_architecture"]
        match model_architecture:
            case "ModelAttackerLSTMLinear":
                return self._get_lstm_linear_model(model_id)
            case "ModelAttackerConvLSTMLinear":
                return self._get_conv_lstm_linear_model(model_id)
            case _:
                raise ValueError(f"Model architecture {model_architecture} not supported.")

    def _add_lstm_linear_model(self,
                               model_id: str,
                               data: DatasetAttackerLSTM,
                               loader: DataLoaderAttackerLSTM,
                               model: ModelAttackerLSTM,
                               trainer: TrainerAttackerLSTM,
                               tester: TesterAttackerLSTM) -> None:
        """
        Add an LSTM linear model to the DataFrame.

        :param model_id: Unique identifier for the model.
        :param data: Dataset used by the model.
        :param loader: DataLoader used by the model.
        :param model: Model instance.
        :param trainer: Trainer instance.
        :param tester: Tester instance.
        """
        self.models.loc[model_id] = pd.Series({
            "data_structure": data.__class__.__name__,
            "data_num_snps": data.num_snps,
            "loader_genome_batch_size": loader.genome_batch_size,
            "loader_snp_batch_size": loader.snp_batch_size,
            "model_architecture": model.__class__.__name__,
            "model_conv_in_channels": pd.NA,
            "model_conv_out_channels": pd.NA,
            "model_conv_kernel_size": pd.NA,
            "model_conv_stride": pd.NA,
            "model_lstm_input_size": model.input_size,
            "model_lstm_hidden_size": model.hidden_size,
            "model_lstm_num_layers": model.num_layers,
            "model_lstm_bidirectional": model.bidirectional,
            "model_lstm_dropout": model.dropout,
            "model_linear_out_features": model.num_output,
            "trainer_num_epochs": trainer.num_epochs,
            "trainer_learning_rate": trainer.learning_rate,
            "trainer_best_eval_epoch": trainer.best_eval_epoch,
            "trainer_best_eval_loss": trainer.best_eval_loss,
            "trainer_best_eval_acc": trainer.best_eval_acc,
            "tester_loss": tester.loss,
            "tester_acc": tester.accuracy_score,
            "tester_precision": tester.precision_score,
            "tester_recall": tester.recall_score,
            "tester_f1_score": tester.f1_score,
            "tester_auroc": tester.auroc_score
        })
        match data.__class__.__name__:
            case "DatasetAttackerLSTMBeacon":
                data_structure = "beacon"
            case "DatasetAttackerLSTMPool":
                data_structure = "pool"
            case _:
                raise ValueError(f"Unsupported data structure {data.__class__.__name__}.")
        model_name = f"attacker_lstm_linear_{data_structure}_{model_id}"
        model.save(model_dir=self.models_dir, model_name=model_name)
        self.models.to_csv(os.path.join(self.models_dir, "models.csv"))

    def _add_conv_lstm_linear_model(self,
                                    model_id: str,
                                    data: DatasetAttackerLSTM,
                                    loader: DataLoaderAttackerLSTM,
                                    model: ModelAttackerLSTM,
                                    trainer: TrainerAttackerLSTM,
                                    tester: TesterAttackerLSTM) -> None:
        """
        Add a ConvLSTM linear model to the DataFrame.

        :param model_id: Unique identifier for the model.
        :param data: Dataset used by the model.
        :param loader: DataLoader used by the model.
        :param model: Model instance.
        :param trainer: Trainer instance.
        :param tester: Tester instance.
        """
        self.models.loc[model_id] = pd.Series({
            "data_structure": data.__class__.__name__,
            "data_num_snps": data.num_snps,
            "loader_genome_batch_size": loader.genome_batch_size,
            "loader_snp_batch_size": loader.snp_batch_size,
            "model_architecture": model.__class__.__name__,
            "model_conv_in_channels": model.conv_in_channels,
            "model_conv_out_channels": model.conv_out_channels,
            "model_conv_kernel_size": model._conv_kernel_size,
            "model_conv_stride": model._conv_stride,
            "model_lstm_input_size": model._lstm_input_size,
            "model_lstm_hidden_size": model._lstm_hidden_size,
            "model_lstm_num_layers": model.lstm_num_layers,
            "model_lstm_bidirectional": model._lstm_bidirectional,
            "model_lstm_dropout": model.lstm_dropout_p,
            "model_linear_out_features": model.linear_output_size,
            "trainer_num_epochs": trainer.num_epoches_trained,
            "trainer_learning_rate": trainer.learning_rate,
            "trainer_best_eval_epoch": trainer.best_eval_loss_epoch,
            "trainer_best_eval_loss": trainer.best_eval_loss,
            "trainer_best_eval_acc": trainer.best_eval_accuracy,
            "tester_loss": tester.loss,
            "tester_acc": tester.accuracy_score,
            "tester_precision": tester.precision_score,
            "tester_recall": tester.recall_score,
            "tester_f1_score": tester.f1_score,
            "tester_auroc": tester.auroc_score
        })
        match data.__class__.__name__:
            case "DatasetAttackerLSTMBeacon":
                data_structure = "beacon"
            case "DatasetAttackerLSTMPool":
                data_structure = "pool"
            case _:
                raise ValueError(f"Unsupported data structure {data.__class__.__name__}.")
        model_name = f"attacker_conv_lstm_linear_{data_structure}_{model_id}"
        model.save(model_dir=self.models_dir, model_name=model_name)
        self.models.to_csv(os.path.join(self.models_dir, "models.csv"))

    def _get_lstm_linear_model(self, model_id: str) -> ModelAttackerLSTM:
        """
        Retrieve an LSTM linear model based on its ID.

        :param model_id: Unique identifier for the model.
        :return: Model instance.
        """
        model = ModelAttackerLSTMLinear(lstm_input_size=self.models.loc[model_id, "model_lstm_input_size"],
                                        lstm_hidden_size=self.models.loc[model_id, "model_lstm_hidden_size"],
                                        lstm_num_layers=self.models.loc[model_id, "model_lstm_num_layers"],
                                        lstm_bidirectional=self.models.loc[model_id, "model_lstm_bidirectional"],
                                        lstm_dropout=self.models.loc[model_id, "model_lstm_dropout"],
                                        linear_out_features=self.models.loc[model_id, "model_linear_out_features"])
        data_structure = self.models.loc[model_id, "data_structure"]
        state_dict_name = f"attacker_lstm_linear_{data_structure}_{model_id}"
        model.load(model_dir=self.models_dir, model_name=state_dict_name)
        return model

    def _get_conv_lstm_linear_model(self, model_id: str) -> ModelAttackerLSTM:
        """
        Retrieve a Conv LSTM linear model based on its ID.

        :param model_id: Unique identifier for the model.
        :return: Model instance.
        """
        model = ModelAttackerConvLSTMLinear(conv_in_channels=self.models.loc[model_id, "model_conv_in_channels"],
                                            conv_out_channels=self.models.loc[model_id, "model_conv_out_channels"],
                                            conv_kernel_size=self.models.loc[model_id, "model_conv_kernel_size"],
                                            conv_stride=self.models.loc[model_id, "model_conv_stride"],
                                            lstm_hidden_size=self.models.loc[model_id, "model_lstm_hidden_size"],
                                            lstm_num_layers=self.models.loc[model_id, "model_lstm_num_layers"],
                                            lstm_bidirectional=self.models.loc[model_id, "model_lstm_bidirectional"],
                                            lstm_dropout=self.models.loc[model_id, "model_lstm_dropout"],
                                            linear_out_features=self.models.loc[model_id, "model_linear_out_features"])
        data_structure = self.models.loc[model_id, "data_structure"]
        state_dict_name = f"attacker_conv_lstm_linear_{data_structure}_{model_id}"
        model.load(model_dir=self.models_dir, model_name=state_dict_name)
        return model