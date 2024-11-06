import os.path
from os import PathLike

import pandas as pd

from utils_attacker_lstm.data import DatasetAttackerLSTM, DataLoaderAttackerLSTM
from . import ModelAttackerConvLSTMLinear, TrainerAttackerLSTM, TesterAttackerLSTM

# Headers for the models DataFrame
_models_headers = [
    "id",

    "random_seed",

    "data_structure",
    "data_num_snps",

    "loader_genome_batch_size",
    "loader_snp_batch_size",

    "model_conv_num_layers",
    "model_conv_channel_size",
    "model_conv_kernel_size",
    "model_conv_stride",
    "model_conv_dilation",
    "model_conv_groups",
    "model_conv_activation",
    "model_conv_activation_kwargs",
    "model_conv_dropout_p",
    "model_conv_dropout_first",
    "model_conv_batch_norm",
    "model_conv_batch_norm_momentum",
    "model_conv_lstm_activation",
    "model_conv_lstm_activation_kwargs",
    "model_conv_lstm_dropout_p",
    "model_conv_lstm_dropout_first",
    "model_conv_lstm_layer_norm",
    "model_lstm_num_layers",
    "model_lstm_input_size",
    "model_lstm_hidden_size",
    "model_lstm_proj_size",
    "model_lstm_bidirectional",
    "model_lstm_dropout_p",
    "model_lstm_dropout_first",
    "model_lstm_layer_norm",
    "model_lstm_linear_dropout_p",
    "model_lstm_linear_dropout_first",
    "model_lstm_linear_batch_norm",
    "model_lstm_linear_batch_norm_momentum",
    "model_linear_num_layers",
    "model_linear_num_features",
    "model_linear_activation",
    "model_linear_activation_kwargs",
    "model_linear_dropout_p",
    "model_linear_dropout_first",
    "model_linear_batch_norm",
    "model_linear_batch_norm_momentum",

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
                  random_seed: int,
                  data: DatasetAttackerLSTM,
                  loader: DataLoaderAttackerLSTM,
                  model: ModelAttackerConvLSTMLinear,
                  trainer: TrainerAttackerLSTM,
                  tester: TesterAttackerLSTM) -> None:
        """
        Add a model to the DataFrame.

        :param model_id: Unique identifier for the model.
        :param random_seed: Random seed used by the model.
        :param data: Dataset used by the model.
        :param loader: DataLoader used by the model.
        :param model: Model instance.
        :param trainer: Trainer instance.
        :param tester: Tester instance.
        """
        match data.__class__.__name__:
            case "DatasetAttackerLSTMBeacon":
                data_structure = "beacon"
            case "DatasetAttackerLSTMPool":
                data_structure = "pool"
            case _:
                raise ValueError(f"Unsupported data structure {data.__class__.__name__}.")

        self.models.loc[model_id] = pd.Series({
            "id": model_id,

            "random_seed": random_seed,

            "data_structure": data_structure,
            "data_num_snps": data.num_snps,

            "loader_genome_batch_size": loader.genome_batch_size,
            "loader_snp_batch_size": loader.snp_batch_size,

            "model_conv_num_layers": model.conv_num_layers,
            "model_conv_channel_size": model.conv_channel_size,
            "model_conv_kernel_size": model.conv_kernel_size,
            "model_conv_stride": model.conv_stride,
            "model_conv_dilation": model.conv_dilation,
            "model_conv_groups": model.conv_groups,
            "model_conv_activation": model.conv_activation,
            "model_conv_activation_kwargs": model.conv_activation_kwargs,
            "model_conv_dropout_p": model.conv_dropout_p,
            "model_conv_dropout_first": model.conv_dropout_first,
            "model_conv_batch_norm": model.conv_batch_norm,
            "model_conv_batch_norm_momentum": model.conv_batch_norm_momentum,
            "model_conv_lstm_activation": model.conv_lstm_activation,
            "model_conv_lstm_activation_kwargs": model.conv_lstm_activation_kwargs,
            "model_conv_lstm_dropout_p": model.conv_lstm_dropout_p,
            "model_conv_lstm_dropout_first": model.conv_lstm_dropout_first,
            "model_conv_lstm_layer_norm": model.conv_lstm_layer_norm,
            "model_lstm_num_layers": model.lstm_num_layers,
            "model_lstm_input_size": model.lstm_input_size,
            "model_lstm_hidden_size": model.lstm_hidden_size,
            "model_lstm_proj_size": model.lstm_proj_size,
            "model_lstm_bidirectional": model.lstm_bidirectional,
            "model_lstm_dropout_p": model.lstm_dropout_p,
            "model_lstm_dropout_first": model.lstm_dropout_first,
            "model_lstm_layer_norm": model.lstm_layer_norm,
            "model_lstm_linear_dropout_p": model.lstm_linear_dropout_p,
            "model_lstm_linear_dropout_first": model.lstm_linear_dropout_first,
            "model_lstm_linear_batch_norm": model.lstm_linear_batch_norm,
            "model_lstm_linear_batch_norm_momentum": model.lstm_linear_batch_norm_momentum,
            "model_linear_num_layers": model.linear_num_layers,
            "model_linear_num_features": model.linear_num_features,
            "model_linear_activation": model.linear_activation,
            "model_linear_activation_kwargs": model.linear_activation_kwargs,
            "model_linear_dropout_p": model.linear_dropout_p,
            "model_linear_dropout_first": model.linear_dropout_first,
            "model_linear_batch_norm": model.linear_batch_norm,
            "model_linear_batch_norm_momentum": model.linear_batch_norm_momentum,

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

        model_name = f"model_attacker_{data_structure}_{model_id}"
        model.save(model_dir=self.models_dir, model_name=model_name)

        self.models.to_csv(os.path.join(self.models_dir, "models.csv"))

    def get_model(self, model_id: str) -> ModelAttackerConvLSTMLinear:
        """
        Retrieve a model based on its ID.

        :param model_id: Unique identifier for the model.
        :return: Model instance.
        """
        if model_id not in self.models.index:
            raise ValueError(f"Model ID {model_id} not found.")
        row = self.models.loc[model_id]
        model = ModelAttackerConvLSTMLinear(conv_num_layers=row["model_conv_num_layers"],
                                            conv_channel_size=row["model_conv_channel_size"],
                                            conv_kernel_size=row["model_conv_kernel_size"],
                                            conv_stride=row["model_conv_stride"],
                                            conv_dilation=row["model_conv_dilation"],
                                            conv_groups=row["model_conv_groups"],
                                            conv_activation=row["model_conv_activation"],
                                            conv_activation_kwargs=row["model_conv_activation_kwargs"],
                                            conv_dropout_p=row["model_conv_dropout_p"],
                                            conv_dropout_first=row["model_conv_dropout_first"],
                                            conv_batch_norm=row["model_conv_batch_norm"],
                                            conv_batch_norm_momentum=row["model_conv_batch_norm_momentum"],
                                            conv_lstm_activation=row["model_conv_lstm_activation"],
                                            conv_lstm_activation_kwargs=row["model_conv_lstm_activation_kwargs"],
                                            conv_lstm_dropout_p=row["model_conv_lstm_dropout_p"],
                                            conv_lstm_dropout_first=row["model_conv_lstm_dropout_first"],
                                            conv_lstm_layer_norm=row["model_conv_lstm_layer_norm"],
                                            lstm_num_layers=row["model_lstm_num_layers"],
                                            lstm_input_size=row["model_lstm_input_size"],
                                            lstm_hidden_size=row["model_lstm_hidden_size"],
                                            lstm_proj_size=row["model_lstm_proj_size"],
                                            lstm_bidirectional=row["model_lstm_bidirectional"],
                                            lstm_dropout_p=row["model_lstm_dropout_p"],
                                            lstm_dropout_first=row["model_lstm_dropout_first"],
                                            lstm_layer_norm=row["model_lstm_layer_norm"],
                                            lstm_linear_dropout_p=row["model_lstm_linear_dropout_p"],
                                            lstm_linear_dropout_first=row["model_lstm_linear_dropout_first"],
                                            lstm_linear_batch_norm=row["model_lstm_linear_batch_norm"],
                                            lstm_linear_batch_norm_momentum=row["model_lstm_linear_batch_norm_momentum"],
                                            linear_num_layers=row["model_linear_num_layers"],
                                            linear_num_features=row["model_linear_num_features"],
                                            linear_activation=row["model_linear_activation"],
                                            linear_activation_kwargs=row["model_linear_activation_kwargs"],
                                            linear_dropout_p=row["model_linear_dropout_p"],
                                            linear_dropout_first=row["model_linear_dropout_first"],
                                            linear_batch_norm=row["model_linear_batch_norm"],
                                            linear_batch_norm_momentum=row["model_linear_batch_norm_momentum"])
        model.load(model_dir=self.models_dir, model_name=f"model_attacker_{row['data_structure']}_{model_id}")
        return model
