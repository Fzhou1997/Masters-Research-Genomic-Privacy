import torch
from torch import nn

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_rsids, latent_size=400):
        super().__init__()
        self.latent_size = latent_size
        self.users_embedding = nn.Embedding(num_users, latent_size, sparse=True)
        self.rsids_embedding = nn.Embedding(num_rsids, latent_size, sparse=True)

    def forward(self, user, rsid):
        return self.users_embedding(user) @ self.rsids_embedding(rsid)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(user, rsid)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



if __name__ == "__main__":
    # define model
    mf = MatrixFactorization(10, 20).to(device)

    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mf.parameters(), lr=1e-3)

