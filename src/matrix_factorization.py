import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_rsids, num_features=512):
        super().__init__()
        self.latent_size = num_features
        self.user_embeddings = nn.Embedding(num_users, num_features)
        self.rsid_embeddings = nn.Embedding(num_rsids, num_features)

    def forward(self, user, rsid):
        return (self.user_embeddings(user) * self.rsid_embeddings(rsid)).sum(dim=1)


def get_genome_mask(genome_array):
    mask = genome_array != -1
    return torch.tensor(genome_array, dtype=torch.float), torch.tensor(mask, dtype=torch.bool)


def train(model, optimizer, criterion, num_epochs, genome_tensor, mask_tensor, batch_size=64):
    dataloader = DataLoader(TensorDataset(genome_tensor, mask_tensor), batch_size=batch_size)
    model.train()
    for epoch in range(num_epochs):
        total_loss = torch.tensor(0, device=device, dtype=torch.float)
        total_correct = torch.tensor(0, device=device, dtype=torch.int)
        total_samples = torch.tensor(0, device=device, dtype=torch.int)
        for genome_batch, mask_batch in dataloader:
            genome_batch = genome_batch.to(device)
            mask_batch = mask_batch.to(device)
            users, rsids = mask_batch.nonzero(as_tuple=True)
            predicted = model(users, rsids)
            actual = genome_batch[users, rsids]
            loss = criterion(predicted, actual)
            total_loss += loss.item() * len(users)
            total_correct += (torch.round(predicted) == actual).sum()
            total_samples += len(users)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct.item() / total_samples.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss.item():.4f}, Accuracy: {epoch_accuracy:.4f}')


def evaluate(model, genome_tensor, mask_tensor, batch_size=64):
    dataloader = DataLoader(TensorDataset(genome_tensor, mask_tensor), batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        num_correct = torch.tensor(0, device=device, dtype=torch.int)
        num_total = torch.tensor(0, device=device, dtype=torch.int)
        for genome_batch, mask_batch in dataloader:
            genome_batch = genome_batch.to(device)
            mask_batch = mask_batch.to(device)
            users, rsids = mask_batch.nonzero(as_tuple=True)
            predicted = model(users, rsids)
            actual = genome_batch[users, rsids]
            num_correct += (torch.round(predicted) == actual).sum()
            num_total += mask_batch.sum()
        accuracy = num_correct.item() / num_total.item()
    return accuracy


def impute(model, genome_tensor, mask_tensor):
    num_users, num_rsids = genome_tensor.shape
    with torch.no_grad():
        predicted = model(torch.arange(num_users), torch.arange(num_rsids))
        predicted = torch.round(predicted).clamp(min=0, max=2)
        predicted[~mask_tensor] = genome_tensor[~mask_tensor]
    return predicted


if __name__ == '__main__':
    genome_array = np.load('../data/opensnp/genotype/npy/G_build37_autosomal.npy')
    num_users, num_rsids = genome_array.shape
    num_features = 384
    model = MatrixFactorizationModel(num_users, num_rsids, num_features).to(device)
    genome_tensor, mask_tensor = get_genome_mask(genome_array)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    numb_epochs = 32
    train(model, optimizer, criterion, numb_epochs, genome_tensor, mask_tensor, batch_size=32)
    accuracy = evaluate(model, genome_tensor, mask_tensor, batch_size=32)
    print(f'Accuracy: {accuracy:.4f}')
    #imputed_genome = impute(model, genome_tensor, mask_tensor)
