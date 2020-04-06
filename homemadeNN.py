"""Homemade artificial neural network.

To do:
- research gradient descent improvements
- generalize cost function
- generalize activation function
- change w_mats to self.w_mats in all functions
- adapted learning rate
- relaxation of weights
"""
import numpy as np
import time

class HomemadeNN(object):
    """Homemade artificial neural network."""

    def __init__(self, size_list, seed=-1):
        """Initialize neural network and random weights.

        Args:
            size_list: Number of neurons per layer.
            seed: Seed for numpy.random generator.
        """

        # define NN size
        self.s = size_list
        self.K = size_list[-1]
        self.L = len(self.s)

        # set random seed
        self.seed = seed
        if self.seed >= 0:
            np.random.seed(self.seed)
    
        # initialize random weights in range of -eps to eps    
        self.w_mats = []
        for i in range(self.L-1):
        
            random_mat = np.random.rand(self.s[i+1],self.s[i]+1)
            eps = np.sqrt(6) / np.sqrt(self.s[i] + self.s[i+1])

            weight_mat = random_mat * 2 * eps - eps
            
            self.w_mats.append(weight_mat)

    def act_fun(self, z_mat):
        """Calculate element-wise sigmoid activation function."""
        
        return (1 / (1 + np.exp(-1 * np.array(z_mat))))


    def forward_prop(self, x_mat, w_mats, layer=-1):
        """ Do forward propagation.

        Args:
            x_mat: Input data of shape (data points)x(input layer).
            w_mats: Neural networks weights for each layer.
            layer: Do forward prop only up to indicated layer index.

        Returns:
            a_list: Activation values for each layer including bias.
        """
        
        # do full propagation if no layer index indicated
        if layer == -1:
            layer = len(w_mats)

        # ensure two-dim. matrix shape
        if len(x_mat.shape) == 1:
            x_mat = x_mat[np.newaxis]

        # input layer: no activation
        a_list = []
        a = np.insert(x_mat, 0, 1, axis=1)
        a_list.append(a)

        # loop over remaining layers
        for i in range(layer):

            # select weight matrix
            w_mat = w_mats[i]

            # matrix multiplication
            z = a.dot(w_mat.T)

            # activation function
            a = self.act_fun(z)
        
            # add bias except for output layer
            if i < len(w_mats)-1:
                a = np.insert(a, 0, 1, axis=1)

            # add to list
            a_list.append(a)

        return a_list


    def backward_prop(self, x_data, y_data, w_mats, lam):
        """ Do backward propagation.

        Args:
            x_data: Input data of shape (data points)x(input layer).
            y_data: Label data of shape (data points)x(output layer).
            w_mats: Neural networks weights for each layer.
            lam: Regularization parameter.

        Returns:
            D_mats: Derivative of cost w.r.t. weights for each layer.
        """
    
        # initialize matrices
        delta_mats = []
        d = []
        for i in range(len(w_mats)):
            delta_mats.append(np.zeros(w_mats[i].shape))
            d.append(np.zeros((x_data.shape[0],w_mats[i].shape[1])))
        D_mats = delta_mats[:]

        # do forward propagation to get all activations
        a = self.forward_prop(x_data, w_mats)

        # calculate last d (simple because derivative of cost and activation function match)
        d.append(a[-1] - y_data) #TODO: change here to generalize for different cost and activation functions

        for l in reversed(range(len(w_mats))):

            # select weight matrix
            w_mat = w_mats[l]

            # calculate d vector of layer
            wd = d[l+1].dot(w_mat)
            d[l] = wd * a[l] * (1 - a[l]) #TODO: change here to generalize for different cost and activation functions

            # lose d values for bias nodes
            d[l] = d[l][:,1:]

            # calculate delta matrix
            delta_mats[l] = d[l+1].T.dot(a[l])

            # add regularization
            reg_w_mat = np.insert(w_mat[:,1:], 0, 0, axis=1)
            D_mats[l] = 1 / x_data.shape[0] * (delta_mats[l] + lam * reg_w_mat)

        return D_mats


    def cost_fun(self, x_data, y_data, w_mats, lam):
        """ Calculate logarithmic cost function.

        Args:
            x_data: Input data of shape (data points)x(input layer).
            y_data: Label data of shape (data points)x(output layer).
            w_mats: Neural networks weights for each layer.
            lam: Regularization parameter.

        Returns:
            cost: Logarithmic cost function value including regularization.
        """
    
        # calculate cost
        h = self.forward_prop(x_data, w_mats)[-1]
        cost = -1 / x_data.shape[0] * (np.trace(y_data.dot(np.log(h).T)) + np.trace((1 - y_data).dot(np.log(1 - h).T)))

        # add regularization
        reg_cost = 0
        
        for l in range(len(w_mats)):
            w_mat = w_mats[l]

            reg_w_mat = w_mat[:,1:]
            
            reg_cost += np.trace(reg_w_mat.dot(reg_w_mat.T))

        reg_cost *= lam / 2 / x_data.shape[0]
        
        # total cost
        cost += reg_cost
        
        return cost


    def grad_descent(self, w_mats, D_mats, alpha):
        """Perform gradient descent.

        Args:
            w_mats: Neural networks weights for each layer.
            D_mats: Derivative of cost w.r.t. weights for each layer.
            alpha: Learning rate.

        Returns:
            w_mats: Updated NN weights for each layer.
        """

        for l in range(len(w_mats)):
            w_mats[l] = w_mats[l] - alpha * D_mats[l]

        return w_mats


    def shuffle_data(self, x_data, y_data):
        """Shuffle order of data points."""

        # get number of output layer nodes
        s_L = y_data.shape[1]

        # merge data
        data = np.concatenate((y_data, x_data), axis=1)
        
        # shuffle data
        np.random.shuffle(data)
        
        # split data
        x_data = data[:,s_L:]
        y_data = data[:,:s_L]

        return (x_data, y_data)


    def test(self, x_data, y_data):
        """Test neural network performance."""

        return np.sum(np.argmax(self.forward_prop(x_data, self.w_mats)[-1], axis=1) == np.argmax(y_data, axis=1))/y_data.shape[0]


    def predict(self, x_data):
        """Predict output based on NN weights."""

        return np.argmax(self.forward_prop(x_data, self.w_mats)[-1], axis=1)

    
    def train(self, x_data, y_data, alpha, lam, max_iter, eps=1e-6, batch_size=0, shuffle_opt=0, quiet=0):
        """Train neural network.

        Args:
            x_data: Input data of shape (data points)x(input layer).
            y_data: Label data of shape (data points)x(output layer).
            alpha: Learning rate.
            lam: Regularization parameter.
            max_iter: Maximum number of iterations.
            eps: Rel. diff. in cost between to iteration to stop training.
            batch_size: Number of data points to be used at once for training.
            shuffle_opt: Shuffle data points in each iteration.
            quiet: Do not print anything.
        """

        # calculate number of batches
        if batch_size == 0:
            batch_size = x_data.shape[0]

        num_batches = x_data.shape[0] // batch_size
        if x_data.shape[0] % batch_size:
            num_batches += 1

        # unnecessary
        x_train = x_data[:]
        y_train = y_data[:]

        # monitor cost function convergence
        self.conv_data = [self.cost_fun(x_data, y_data, self.w_mats, lam)]

        for i in range(max_iter):

            if not quiet:
                start_time = time.time()    

            # shuffle data
            if shuffle_opt:
                x_train, y_train = self.shuffle_data(x_data, y_data)
        
            # loop over number of batches
            for b in range(num_batches):

                # get batch data
                x_batch = x_train[(b*batch_size):((b+1)*batch_size),:]
                y_batch = y_train[(b*batch_size):((b+1)*batch_size),:]

                # do back propagation
                D_mats = self.backward_prop(x_batch, y_batch, self.w_mats, lam)

                # do gradient descent
                self.w_mats = self.grad_descent(self.w_mats, D_mats, alpha)

            # check convergence
            self.conv_data.append(self.cost_fun(x_data, y_data, self.w_mats, lam))
            if abs((self.conv_data[-2]-self.conv_data[-1])/self.conv_data[-2]) < eps:
                break

            # print information
            if not quiet:            
                stop_time = time.time()
                print("Iteration: %d - Cost: %f - Time: %f - Alpha: %f" % (i, self.conv_data[-1], stop_time - start_time, alpha))
            
                if (i % 10) == 0:
                    print("Accuracy: %.2f %%" % (self.test(x_data, y_data)*100))

        if not quiet:
            print('The End.')
            

    