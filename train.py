import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import functions as fc
import tensorflow as tf
import torch.nn as nn
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from model import Model

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

epochs = 1000
hidden_size = 100
reduced_dim = 500
print("Loading arrays...")
data = pd.read_csv('df.csv')
cls  = pd.read_csv('cls.csv').iloc[: , 1:]

vocab, pipe, corpus = fc.encodeData(data)

X = torch.from_numpy(pipe['count'].transform(corpus).toarray().astype(np.float32))

print("Reducing Large Dimensions...")
pca = PCA(n_components = reduced_dim)
X = torch.from_numpy(np.array(pca.fit_transform(X)).astype(np.float32)).to(device)



print("Mapping Classes...")
embed = {"Bart Simpson": [1, 0, 0, 0, 0],
         "Homer Simpson": [0, 1, 0, 0, 0],
         "Lisa Simpson": [0, 0, 0, 1, 0],
         "Marge Simpson": [0, 0, 0, 0, 1],
         "Other": [0, 0, 1, 0, 0]
}
Y = []
for c in data['class']:
    Y.append(embed[c])

Y = torch.from_numpy(np.array(Y).astype(np.float32)).to(device)

print("Initializing Model")
model = Model(X.shape[1], hidden_size).to(device)
model.train()
print("\nTraining Model...")
loss = fc.train(X, Y, model, epochs)

plt.plot(loss)
plt.savefig('700_epochs.png')

print("Saving Model...")
torch.save(model, "model.tensor")
