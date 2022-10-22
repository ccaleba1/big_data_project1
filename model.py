import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, embedding_dim, hidden_size):
        super(Model, self).__init__()
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_size,
                           num_layers=1,
                           bidirectional=True,
                           batch_first=True)

        #dense layer
        self.hidden = nn.Linear(hidden_size, 5)

        #activation function
        self.activation = nn.Sigmoid()

    def forward(self, text):
        out = text.view(-1, text.shape[1])
        output, (hidden, cell) = self.lstm(out)

        out = self.hidden(output)

        return self.activation(out)
