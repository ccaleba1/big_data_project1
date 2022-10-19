import torch
import wandb
import functions as fc
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from model import Model

from tqdm import tqdm

def getData(count = None):
    with open("unmcs567-project1/simpsons_dataset-training.tsv", 'r') as f:
        lines = f.readlines()
    iter = 0
    list = []
    ids = []
    classifications = []
    subclass = []
    data = []

    for line in lines:
        if(count):
            if(iter == count): break

        list.append(line.split("\t"))
        iter+=1

    list.pop(0)
    for e in list:
        ids.append(e[0])
        classifications.append([e[1], e[2]])
        data.append(e[3])

    vec_data = []
    max_length = 0
    for text in data:
        vec_data.append(text.split())
        length = len(text.split())
        if max_length < length:
            max_length = length

        for text in vec_data:
            if len(text) < max_length:
                text+=[' '] * (max_length - len(text))

    print("Creating Embeddings For Text...\n")
    label_encoder = LabelEncoder()
    vec_data = np.array(vec_data)
    X = []
    for text in vec_data:
        embeddings = label_encoder.fit_transform(text)
        X.append(embeddings)
    X = np.array(X)

    print("Creating Embeddings For Classifications...\n")
    classifications = np.array(classifications)
    Y = []
    for classes in classifications:
        embeddings = label_encoder.fit_transform(classes)
        Y.append(embeddings)
    Y = np.array(Y)

    print("Returning data...\n")
    return np.array(ids), Y, X


def train(ids, X, Y, model, epochs, lr = 0.001):
    loss_plot = []
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    for epoch in tqdm(range(epochs)):
        optim.zero_grad()
        predictions = model(X)
        targets = Y
        loss = loss_func(predictions, targets)
        loss_plot.append(loss.item())
        loss.backward()
        optim.step()
    return loss_plot 
