import torch
import functions as fc
import numpy as np
import torch.nn as nn
import torch
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from tensorflow import keras
import tensorflow_text as text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from model import Model
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf2
tf2.disable_eager_execution()



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
        print("Encoding Data")
        encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        trainable=True)
        outputs = encoder(corpus)
        embeddings = outputs["pooled_output"]
        print(embeddings)
        sequence_output = outputs["sequence_output"]


        return vocab, embeddings, corpus

    corpus = data['text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)

    vocab = {}
    for e in corpus:
        vocab[e] = 1

    print("Encoding Data")

    # Load pre trained ELMo model
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

    # create an instance of ELMo
    embeddings = elmo(
        corpus,
        signature="default",
        as_dict=True)["elmo"]
    init = tf2.initialize_all_variables()
    sess = tf2.Session()
    sess.run(init)

    # Print word embeddings for word WATCH in given two sentences
    print('Word embeddings for word WATCH in first sentence')
    print(sess.run(embeddings[0][3]))
    print('Word embeddings for word WATCH in second sentence')
    print(sess.run(embeddings[1][5]))

    vocab = list(vocab.keys())

    # pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
    #               ('tfid', TfidfTransformer())]).fit(corpus)

    return vocab, embeddings, corpus

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
