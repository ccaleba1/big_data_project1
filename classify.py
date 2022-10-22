import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functions as fc
import torch.nn as nn
from model import Model
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import functions as fc

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
reduced_dim = 100

print("Loading arrays...")
ids, X = fc.getTesting()
print(ids)
data = pd.read_csv('df.csv')

vocab, pipe, corpus = fc.encodeData(X, data)

X = torch.from_numpy(pipe['count'].transform(corpus).toarray().astype(np.float32))

print("Reducing Large Dimensions...")
pca = PCA(n_components = reduced_dim)
X = torch.from_numpy(np.array(pca.fit_transform(X)).astype(np.float32))

print("Mapping Classes...")
embed = {"Bart Simpson": [1., 0., 0., 0., 0.],
         "Homer Simpson": [0., 1., 0., 0., 0.],
         "Lisa Simpson": [0., 0., 0., 1., 0.],
         "Marge Simpson": [0., 0., 0., 0., 1.],
         "Other": [0., 0., 1., 0., 0.]
}

print("Loading Model...\n")
model = torch.load('model.tensor', map_location=torch.device('cpu'))
model.eval()

predictions = []

outputs = model(X)

print("Classifying...")
max_val = outputs.argmax(1)
out = torch.zeros(outputs.shape).scatter (1, max_val.unsqueeze(1), 1)
print("Creating Output File...")
with open("o.csv", 'a') as f:
    convert = list(embed.values())
    f.write("Id,Category\n")
    ids = ids.values.tolist()
    i = 0
    for tens in out:
        for e in embed.keys():
            if list(tens) == embed[e]:
                #cls = e
                f.write(str(ids[i]) + "," + str(e) + "\n")
                i+=1
                print(i)
                break
        if i == len(ids):
            break
