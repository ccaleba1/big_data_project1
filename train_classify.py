import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import functions as fc
import tensorflow as tf
import torch.nn as nn
import pandas as pd
import seaborn as sns
from sklearn import metrics
from numpy import array
from numpy import argmax
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

batch_size = 0.8 #modify to change sizes of training and test sets

print("Loading arrays...")
data = pd.read_csv('df.csv')

corpus = fc.encodeData(data)

arr = np.array(corpus).astype(np.float32)

sz = int(arr.shape[0] * batch_size)
X_train = arr[:sz, :] #training set
X_test = arr[sz:, :] #test set

print("Mapping Classes...")
embed = {"Bart Simpson": 0.0,
         "Homer Simpson": 1.0,
         "Lisa Simpson": 2.0,
         "Marge Simpson": 3.0,
         "Other": 4.0
}

Y = []
for c in data['class']:
    Y.append(embed[c])

Y = np.array(Y).astype(np.float32)
Y_train = data['class'][:sz]
Y_test = data['class'][sz:]
print("Training Model...")
model = LinearSVC()
model.fit(X_train, Y_train)

#### comment lines 61-78 if testing on real data ####

print("Classifying on Test Data")
Y_pred = model.predict(X_test)

conf_mat = confusion_matrix(Y_test, Y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=embed.keys(),
            yticklabels=embed.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVCn", size=16);
plt.savefig('Conf_Matrix.png')

with open("classification_report.txt", 'w') as f:
    f.write('CLASSIFICATIION METRICS\n')
    f.write(metrics.classification_report(Y_test, Y_pred,
                                    target_names= embed.keys()))
print("Done!")


#### uncomment lines below if testing on real test data ####

# print("Loading Test Arrays...")
#
# ids, x_test = fc.getTesting()
#
# x_test, X, Y = fc.encodeTest(x_test, data['text'], Y)
#
# model = LinearSVC()
# model.fit(X, Y)
# print("Predicting...")
# y_pred = model.predict(x_test)
#
# y = []
# embed = {0.0: "Bart Simpson",
#          1.0: "Homer Simpson",
#          2.0: "Lisa Simpson",
#          3.0: "Marge Simpson",
#          4.0: "Other"
# }
#
# for pred in y_pred:
#     y.append(embed[pred])
#
#
# with open("pred.csv", 'a') as f:
#     i = 0
#     f.write("Id,Category\n")
#     for cls in y:
#         f.write(str(ids[i]) + "," + str(y[0]) + '\n')
#         i+=1
#
# print("Done!")
