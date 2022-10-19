import torch
import wandb
import pickle as pkl
import functions as fc
import numpy as np


import torch.nn as nn
from model import Model

epochs = 10
hidden_size = 20

ids, classifications, embeddings = fc.getData()

np.savez("embeddings.npz", a1=embeddings, a2=classifications)
np.savez("ids.npz", a1=ids)
print("Done!")
