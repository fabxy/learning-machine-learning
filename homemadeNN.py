"""Homemade code to train and test 

Todo:
- speed up with matrix multiplication
- ensure proper matrix multiplication everywhere
- clean up
"""

import numpy as np
import pandas as pd
import os
import time

np.random.seed(0)

def act_fun(z_mat):
    return (1 / (1 + np.exp(-1 * np.array(z_mat))))

def forward_prop(x_mat, w_mats, layer=-1):
        
    if layer == -1:
        layer = len(w_mats)

    if len(x_mat.shape) == 1:
        x_mat = x_mat[np.newaxis]

    a_list = []
    a = np.insert(x_mat, 0, 1, axis=1)
    a_list.append(a)

    for i in range(layer):

        # select weight matrix
        w_mat = w_mats[i]

        # 2D dot operation
        z = a.dot(w_mat.T)

        # multiply activation function
        a = act_fun(z)
    
        # add bias expect for output layer
        if i < len(w_mats)-1:
            a = np.insert(a, 0, 1, axis=1)

        # add to list
        a_list.append(a)

    return a_list

def backward_prop(x_data, y_data, w_mats, lam):
    
    delta_mats = []
    d_zero = []
    for i in range(len(w_mats)):
        delta_mats.append(np.zeros(w_mats[i].shape))
        d_zero.append(np.zeros(w_mats[i].shape[1]))
    D_mats = delta_mats[:]

    for t in range(x_data.shape[0]):
        
        d = d_zero[:]

        a = forward_prop(x_data[t,:], w_mats)
        a = a[-1]
        d.append((a - y_data[t]).T)

        for l in reversed(range(len(w_mats))):

            w_mat = w_mats[l]
            wd = w_mat.T.dot(d[l+1])
            
            a = forward_prop(x_data[t], w_mats, l)
            a = a[-1].T

            d[l] = np.array([wd[k] * a[k] * (1-a[k]) for k in range(len(a))])
            d[l] = d[l][1:]

            delta_mats[l] += d[l+1].dot(a.T)

    # add regularization
    for l in range(len(w_mats)):
        w_mat = w_mats[l]
        reg_w_mat = np.insert(w_mat[:,1:], 0, 0, axis=1)
        
        D_mats[l] = 1 / x_data.shape[0] * (delta_mats[l] + lam * reg_w_mat)

    return D_mats

def backward_prop_fast(x_data, y_data, w_mats, lam):
    
    delta_mats = []
    d = []
    for i in range(len(w_mats)):
        delta_mats.append(np.zeros(w_mats[i].shape))
        d.append(np.zeros((x_data.shape[0],w_mats[i].shape[1])))
    D_mats = delta_mats[:]

    a = forward_prop(x_data, w_mats)
    d.append(a[-1] - y_data)

    for l in reversed(range(len(w_mats))):

        w_mat = w_mats[l]

        wd = d[l+1].dot(w_mat)
        d[l] = wd * a[l] * (1 - a[l])

        d[l] = d[l][:,1:]

        delta_mats[l] = d[l+1].T.dot(a[l])

        # add regularization
        reg_w_mat = np.insert(w_mat[:,1:], 0, 0, axis=1)
        D_mats[l] = 1 / x_data.shape[0] * (delta_mats[l] + lam * reg_w_mat)

    return D_mats


def cost_fun(x_data, y_data, w_mats, lam):
    
    m = x_data.shape[0]
    K = w_mats[-1].shape[0]
    
    cost = 0
    for t in range(m):
        for k in range(K):
            cost += y_data[t, k] * np.log(forward_prop(x_data[t], w_mats)[-1][:,k]) + (1 - y_data[t, k]) * np.log(1 - forward_prop(x_data[t], w_mats)[-1][:,k])

    cost *= -1 / m

    # add regularization
    reg_cost = 0

    for l in range(len(w_mats)):
        w_mat = w_mats[l]

        for i in range(1, w_mat.shape[1]):
            for j in range(w_mat.shape[0]):
                reg_cost += w_mat[j,i]**2

    reg_cost *= lam / 2 / m

    return cost + reg_cost

def cost_fun_fast(x_data, y_data, w_mats, lam):
    
    # calculate cost
    h = forward_prop(x_data, w_mats)[-1]
    cost = -1 / x_data.shape[0] * (np.trace(y_data.dot(np.log(h).T)) + np.trace((1 - y_data).dot(np.log(1 - h).T)))

    # add regularization
    reg_cost = 0
    
    for l in range(len(w_mats)):
        w_mat = w_mats[l]

        reg_w_mat = w_mat[:,1:]
        
        reg_cost += np.trace(reg_w_mat.dot(reg_w_mat.T))

    reg_cost *= lam / 2 / x_data.shape[0]

    return cost + reg_cost


def grad_descent(w_mats, D_mats, alpha):

    for l in range(len(w_mats)):
        w_mats[l] = w_mats[l] - alpha * D_mats[l]

    return w_mats
        

if __name__ == "__main__":
    
    ### Neural network parameters

    # # Number of layers including input and output layer
    # L = 3

    # # Number of units in each layer without bias unit
    # s = [4, 4, 1]

    # # Number of units in output layer
    # K = 1

    # What do we need:
    # data, split up in training and testing data, initial weights, activation function

    # What do we want to do?
    # Let's say we have only training example: x with size of (s[0]+1)x1 and y of size 1x1
    # forward propagation: h = a_1_3 = act()

    print(os.getcwd())

    # data: image size 28x28 - training set size 60k
    # data = pd.read_csv('learning-machine-learning/data/mnist/mnist_train.csv')

    # x_data = data.iloc[:,1:]
    # y_data = data.iloc[:,0]

    # data: image size 20x20 - training set size 5k
    x_data = pd.read_csv('learning-machine-learning/data/mnist/mnist_x_data.csv', header=None)
    y_data = pd.read_csv('learning-machine-learning/data/mnist/mnist_y_data.csv', header=None)

    # x_data to numpy array - why loading via pandas at all?
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # mix up data
    data = np.concatenate((y_data, x_data), axis=1)
    np.random.shuffle(data)
    x_data = data[:,1:]
    y_data = data[:,0]

    y_org = y_data[:]

    # y_data to vec form - do as matrix operation?
    y_vecs = np.zeros((len(y_data), int(max(y_data)-min(y_data))+1))

    for i in range(y_vecs.shape[0]):
        y_val = y_data[i]
        if y_val == 10:
            y_val = 0
        y_vecs[i, int(y_val)] = 1
    
    y_data = y_vecs

    # neural network parameters
    s_hidden = [25]
    K = 10
    lam = 10
    alpha = 0.5

    alpha_start = 1.0
    alpha_end = 0.001

    s = [x_data.shape[1]] + s_hidden + [K]
    L = len(s)
    
    # initialize random weights in range of -eps to eps    
    weight_mats = []
    for i in range(L-1):
        
        random_mat = np.random.rand(s[i+1],s[i]+1)
        eps = np.sqrt(6) / np.sqrt(s[i] + s[i+1])

        weight_mat = random_mat * 2 * eps - eps
        
        weight_mats.append(weight_mat)

    # train NN
    i = 1
    cost = cost_fun_fast(x_data, y_data, weight_mats, lam)

    max_iter = 500
    mini_batch = 5000

    for i in range(max_iter):

        start_time = time.time()    

        num_batches = x_data.shape[0] // mini_batch
        if x_data.shape[0] % mini_batch:
            num_batches += 1

        alpha = alpha_start + i * (alpha_end-alpha_start) / max_iter
    
        for b in range(num_batches):

            x_batch = x_data[(b*mini_batch):((b+1)*mini_batch),:]
            y_batch = y_data[(b*mini_batch):((b+1)*mini_batch),:]

            D_mats = backward_prop_fast(x_batch, y_batch, weight_mats, lam)
            weight_mats = grad_descent(weight_mats, D_mats, alpha)
        
        stop_time = time.time()
        cost = cost_fun_fast(x_data, y_data, weight_mats, lam)
        print("Iteration: %d - Cost: %f - Time: %f - Alpha: %f" % (i, cost, stop_time - start_time, alpha))
        
        if (i % 10) == 0:
            res = np.sum(np.argmax(forward_prop(x_data, weight_mats)[-1], axis=1) == y_org)
            print("Accuracy: %.2f %%" % (res/len(y_org)*100))

    # test NN
    res = np.sum(np.argmax(forward_prop(x_data, weight_mats)[-1], axis=1) == y_org)
    print("Accuracy: %.2f %%" % (res/len(y_org)*100))

     

