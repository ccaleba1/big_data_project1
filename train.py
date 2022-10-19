import torch
import wandb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import functions as fc
import torch.nn as nn
from model import Model
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

epochs = 1000
hidden_size = 200

print("Loading arrays...")
arrs = np.load('ids.npz')
ids = arrs['a1']

print("Creating tensors...")
data = np.load('embeddings.npz')
X = torch.from_numpy(data['a1'])
Y = torch.from_numpy(data['a2'])

del arrs
del data

print("Initializing Model")
model = Model(X.shape[1], hidden_size)
model.train()
print("\nTraining Model...")
loss = fc.train(ids, X.float(), Y.float(), model, epochs)

plt.plot(loss)
plt.show()

print("Saving Model...")
torch.save(model.state_dict(), "model.tensor")
