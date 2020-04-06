"""Training homemadeNN on NIST dataset."""

import numpy as np
import pandas as pd
import homemadeNN

# load data
# image size 20x20 - training set size 5k
x_data = pd.read_csv('learning-machine-learning/data/nist/nist_x_data.csv', header=None)
y_data = pd.read_csv('learning-machine-learning/data/nist/nist_y_data.csv', header=None)

# x_data to numpy array
x_data = np.array(x_data)
y_data = np.array(y_data)

# y_data to vec form - do as matrix operation?
y_vecs = np.zeros((len(y_data), int(max(y_data)-min(y_data))+1))

for i in range(y_vecs.shape[0]):
    y_val = y_data[i]
    if y_val == 10:
        y_val = 0
    y_vecs[i, int(y_val)] = 1

y_data = y_vecs

# neural network hyperparameters
s_hidden = [25]

max_iter = 500
alpha = 0.001
lam = 0.01
batch_size = 1
shuffle_opt = 1

# create neural network
ann = homemadeNN.HomemadeNN([x_data.shape[1]] + s_hidden + [y_data.shape[1]], 0)

# shuffle data before training
x_data, y_data = ann.shuffle_data(x_data, y_data)

# train neural network
ann.train(x_data, y_data, alpha, lam, max_iter, batch_size=batch_size, shuffle_opt=shuffle_opt)
