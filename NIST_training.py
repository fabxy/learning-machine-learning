"""Training homemadeNN on NIST dataset."""

import numpy as np
import pandas as pd
import homemadeNN
from sklearn.model_selection import train_test_split

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
s_hidden = [200]

max_iter = 50
alpha = 0.1
lam = 1
batch_size = 10
shuffle_opt = 1
rseed = 5
test_data_size = 0.2

# create neural network
ann = homemadeNN.HomemadeNN([x_data.shape[1]] + s_hidden + [y_data.shape[1]], rseed)

# shuffle and split data before training
x_data, y_data = ann.shuffle_data(x_data, y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data_size, random_state=rseed)

# check fair distribution of classes
class_count = np.zeros(y_data.shape[1])
for y_vec in y_train:
    class_count[np.argmax(y_vec)] += 1
print("Average number of samples per class in training data: %d" % int(np.mean(class_count)))
print("Maximum absolute deviation from average: %d" % int(max(abs(class_count - np.mean(class_count)))))

# train neural network
print("Training neural network:")
ann.train(x_train, y_train, alpha, lam, max_iter, batch_size=batch_size, shuffle_opt=shuffle_opt)

# test neural network
print("Performance on test data:")
print("Accuracy: %.2f %%" % (ann.test(x_test, y_test)*100))
