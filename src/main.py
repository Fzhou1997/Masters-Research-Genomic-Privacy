import multiprocessing as mp

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.hair_color.hair_color_dataloader import HairColorDataLoader
from src.hair_color.hair_color_dataset import HairColorDataset
from src.hair_color.hair_color_lstm_model import HairColorLSTMModel

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

TRAIN_RATIO = 0.8
BATCH_SIZE = 32
INPUT_SIZE = 1
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 3
LEARNING_RATE = 0.0001
NUM_EPOCHS = 32

if __name__ == '__main__':
    mp.freeze_support()
    dataset = HairColorDataset()
    dataset.preprocess()
    dataset.load()
    train_set, test_set = dataset.split_train_test(train_ratio=TRAIN_RATIO)
    train_loader = HairColorDataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, weighted_sampling=True)
    test_loader = HairColorDataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, weighted_sampling=False)
    model = HairColorLSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    model.eval()