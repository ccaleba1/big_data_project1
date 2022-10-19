import torch
import torch.nn as nn

from gensim import utils

class Model(nn.Module):

    def __init__(self, embedding_dim, hidden_size):
        super(Model, self).__init__()

        self.lin1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()

        self.lin2 = nn.Linear(hidden_size, 8)

        self.lin3 = nn.Linear(8, 2)
        self.activation = nn.Sigmoid()

    def forward(self, text):
        out = text.view(-1, text.shape[1])
        out = self.lin1(out.float())
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.lin3(out)
        return self.activation(out)
