import torch
import functions as fc
import numpy as np
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from model import Model
from tqdm import tqdm

def getData():
    data = pd.read_csv("unmcs567-project1/simpsons_dataset-training.tsv",
           sep="\t")

    print("Processing Data...")
    text = data['text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True).str.split(expand=True)

    unique = data['class'].unique()
    cls = pd.get_dummies(
        unique,
    )
    print("Returning Data...")
    return data, text, cls

def encodeData(data, sec_data=pd.DataFrame()):

    if not sec_data.empty:
        corpus = sec_data['text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)
        vocab = {}
        for e in corpus:
            for word in e.split():
                vocab[word] = 1

        vocab = list(vocab.keys())

        pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                      ('tfid', TfidfTransformer())]).fit(data)
        return vocab, pipe, corpus

    corpus = data['text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)
    vocab = {}
    for e in corpus:
        for word in e.split():
            vocab[word] = 1

    vocab = list(vocab.keys())

    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                  ('tfid', TfidfTransformer())]).fit(corpus)

    return vocab, pipe, corpus

def train(X, Y, model, epochs, lr = 0.001):
    loss_plot = []
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()

    for epoch in tqdm(range(epochs)):
        optim.zero_grad()
        predictions = model(X)
        loss = loss_func(predictions, Y)
        loss_plot.append(loss.item())
        loss.backward()
        optim.step()
    return loss_plot

def getTesting():
    data = pd.read_csv("unmcs567-project1/simpsons_dataset-testing.tsv",
           sep="\t")

    print("Processing Data...")
    X = data['text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)
    id = data['id']
    return id, X
