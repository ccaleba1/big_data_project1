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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from mod import Model
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf2
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def getData():
    data = pd.read_csv("unmcs567-project1/simpsons_dataset-training.tsv",
           sep="\t")

    print("Processing Data...")
    text = data['text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)

    print("Returning Data...")
    return data, text

def encodeData(data):
    #text is now lowercase and each row is a non-tokenized sentence
    corpus = data['text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)

    print("Encoding Data")

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')

    #transform each sentence into a vector
    features = tfidf.fit_transform(corpus).toarray()

    return features

def encodeTest(x_test, corpus, Y):
    #function used to encode test data for final classification
    corpus = corpus.str.lower().str.replace(r'[^\w\s]+', '', regex=True)
    print("Encoding Data")

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')

    #transform each complaint into a vector
    Y = Y[:x_test.shape[0]] #uncomment if testing on validation set
    features = tfidf.fit(corpus.iloc[:x_test.shape[0]]) #uncomment if testing on validation set
    fitted_features = features.transform(corpus.iloc[:x_test.shape[0]]) #uncomment if testing on validation set

    #transform each complaint into a vector
    #Y=Y #comment if testing on validation set
    #features = tfidf.fit(corpus) #comment if testing on validation set
    #fitted_features = features.transform(corpus) #comment if testing on validation set

    #returning a fitted testing set that matches model params of training set
    return features.transform(x_test), fitted_features, Y

def getTesting():
    data = pd.read_csv("unmcs567-project1/simpsons_dataset-testing.tsv",
           sep="\t")

    print("Processing Data...")
    X = data['text'].str.lower().str.replace(r'[^\w\s]+', '', regex=True)
    id = data['id']
    return id, X
