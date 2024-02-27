import numpy as np

import torch
from torch import nn

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_rsids, num_features=400):
        super().__init__()
        self.latent_size = num_features
        self.user_embeddings = nn.Embedding(num_users, num_features, sparse=True)
        self.rsid_embeddings = nn.Embedding(num_rsids, num_features, sparse=True)

    def forward(self, user, rsid):
        user_embedding = self.user_embeddings(user)
        rsid_embedding = self.rsid_embeddings(rsid)
        return (user_embedding * rsid_embedding).sum(dim=1)


def get_genome_mask(genome_array):
    mask = genome_array != -1
    return torch.tensor(genome_array, dtype=torch.float), torch.tensor(mask, dtype=torch.bool)


def train(model, optimizer, criterion, num_epochs, genome_tensor, mask):
    for epoch in range(num_epochs):
        user_idx, item_idx = np.where(mask)
        user_idx = torch.tensor(user_idx)
        item_idx = torch.tensor(item_idx)

        predicted = model(user_idx, item_idx)
        actual = genome_tensor[user_idx, item_idx]

        loss = criterion(predicted, actual)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')


def impute(model, genome_tensor, mask):
    num_users, num_rsids = genome_tensor.shape
    with torch.no_grad():
        predicted = model(torch.arange(num_users), torch.arange(num_rsids))
        predicted = torch.round(predicted).clamp(min=0, max=2)
        predicted = predicted.type(torch.int)
        predicted[~mask] = genome_tensor[~mask]
    return predicted


if __name__ == '__main__':
    genome_array = np.load('../data/opensnp/genotype/npy/G_build37_autosomal.npy')
    num_users, num_rsids = genome_array.shape
    num_features = 400
    model = MatrixFactorizationModel(num_users, num_rsids, num_features).to(device)
    genome_tensor, mask = get_genome_mask(genome_array)
    genome_tensor = genome_tensor.to(device)
    mask = mask.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    num_epoches = 10
    train(model, optimizer, criterion, num_epoches, genome_tensor, mask)
    imputed_genome = impute(model, genome_tensor, mask)
