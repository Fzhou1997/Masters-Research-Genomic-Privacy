import multiprocessing as mp

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split
from tqdm import tqdm

from src.hair_color.hair_color_dataloader import HairColorDataLoader
from src.hair_color.hair_color_dataset import HairColorDataset
from src.hair_color.hair_color_lstm_model import HairColorLSTMModel

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == '__main__':
    mp.freeze_support()
    train_ratio = 0.8
    batch_size = 32
    input_size = 1
    hidden_size = 128
    num_layers = 2
    num_classes = 3
    learning_rate = 0.0001
    num_epochs = 32

    dataset = HairColorDataset(37)
    dataset.load('../data/genomes/hair_color')
    dataset.to_device(device)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = HairColorDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = HairColorDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = HairColorLSTMModel(input_size, hidden_size, num_layers, num_classes)
    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print(dataset[0])

    model.train()
    for epoch in range(num_epochs):
        for genotype, phenotype in train_loader:
            genotype = genotype.to(device)
            phenotype = phenotype.to(device)
            outputs = model(genotype)
            loss = criterion(outputs, phenotype)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for genotype, phenotype in test_loader:
            genotype = genotype.to(device)
            phenotype = phenotype.to(device)
            outputs = model(genotype)
            _, predicted = outputs.data
            total += phenotype.size(0)
            correct += (predicted == phenotype).sum().item()
        print(f'Accuracy: {100 * correct / total}%')
