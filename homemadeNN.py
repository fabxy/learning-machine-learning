"""Homemade artificial neural network for classification."""
import numpy as np
import time

class HomemadeNN(object):
    """Homemade artificial neural network for classification."""

    def __init__(self, size_list, act_funs='sigmoid', seed=-1):
        """Initialize neural network and random weights.

        Args:
            size_list: Number of neurons per layer.
            act_funs: Activation function name per layer.
            seed: Seed for numpy.random generator.
        """

        # define NN size
        self.s = size_list
        self.K = size_list[-1]
        self.L = len(self.s)

        # set activation function for each layer
        if isinstance(act_funs, list):
            self.afs = act_funs
        else:
            self.afs = [act_funs for _ in range(self.L)]

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


    def act_fun(self, z_mat, fun_name):
        """Calculate element-wise activation function.
        
        Implemented: sigmoid, tanh, ReLU.

        Args:
            z_mat: Input matrix to activation function.
            fun_name: String of activation function name.

        Returns:
            Output matrix with element-wise applied activation function.
        """
        
        # sigmoid function 
        if fun_name == 'sigmoid':
            return (1 / (1 + np.exp(-1 * np.array(z_mat))))
        
        # hyperbolic tangent
        elif fun_name == 'tanh':
            return np.tanh(z_mat)
        
        # rectified linear unit
        elif fun_name == 'ReLU':
            return np.maximum(0, z_mat)

    
    def der_act_fun(self, z_mat, fun_name):
        """Calculate element-wise derivate of activation function."""

        # sigmoid function 
        if fun_name == 'sigmoid':
            a = self.act_fun(z_mat, fun_name)
            return a * (1 - a)
        
        # hyperbolic tangent
        elif fun_name == 'tanh':
            return 1 - (np.tanh(z_mat))**2
        
        # rectified linear unit
        elif fun_name == 'ReLU':
            return np.where(z_mat >= 0, 1.0, 0.0)


    def forward_prop(self, x_mat, layer=-1):
        """ Do forward propagation.

        Args:
            x_mat: Input data of shape (data points)x(input layer).
            layer: Do forward prop only up to indicated layer index.

        Returns:
            a_list: Activation values for each layer including bias.
            z_list: Input values for each layer excluding bias.
        """
        
        # do full propagation if no layer index indicated
        if layer == -1:
            layer = len(self.w_mats)

        # ensure two-dim. matrix shape
        if len(x_mat.shape) == 1:
            x_mat = x_mat[np.newaxis]

        # input layer: no activation
        a_list = []
        a = np.insert(x_mat, 0, 1, axis=1)
        a_list.append(a)
        z_list = [x_mat]

        # loop over remaining layers
        for l in range(layer):

            # select weight matrix
            w_mat = self.w_mats[l]

            # matrix multiplication
            z = a.dot(w_mat.T)

            # add to list
            z_list.append(z)

            # activation function
            a = self.act_fun(z, self.afs[l+1])
        
            # add bias except for output layer
            if l < len(self.w_mats)-1:
                a = np.insert(a, 0, 1, axis=1)

            # add to list
            a_list.append(a)

        return a_list, z_list


    def backward_prop(self, x_data, y_data, lam):
        """ Do backward propagation.

        Args:
            x_data: Input data of shape (data points)x(input layer).
            y_data: Label data of shape (data points)x(output layer).
            lam: Regularization parameter.

        Returns:
            D_mats: Derivative of cost w.r.t. weights for each layer.
        """
    
        # initialize matrices
        delta_mats = []
        d = []
        for i in range(len(self.w_mats)):
            delta_mats.append(np.zeros(self.w_mats[i].shape))
            d.append(np.zeros((x_data.shape[0],self.w_mats[i].shape[1])))
        D_mats = delta_mats[:]

        # do forward propagation to get all activations
        a, z = self.forward_prop(x_data)

        # calculate last d
        d.append(self.der_cost_fun(y_data, a[-1]) * self.der_act_fun(z[-1], self.afs[-1]))

        for l in reversed(range(len(self.w_mats))):

            # select weight matrix
            w_mat = self.w_mats[l]

            # calculate d vector of layer
            wd = d[l+1].dot(w_mat)
            wd = wd[:,1:] # exclude bias nodes

            d[l] = wd * self.der_act_fun(z[l], self.afs[l])

            # calculate delta matrix
            delta_mats[l] = d[l+1].T.dot(a[l])

            # add regularization
            reg_w_mat = np.insert(w_mat[:,1:], 0, 0, axis=1)
            D_mats[l] = 1 / x_data.shape[0] * (delta_mats[l] + lam * reg_w_mat)

        return D_mats


    def cost_fun(self, x_data, y_data, lam):
        """ Calculate cost function.

        Implemented: Logarithmic cost function.

        Args:
            x_data: Input data of shape (data points)x(input layer).
            y_data: Label data of shape (data points)x(output layer).
            lam: Regularization parameter.

        Returns:
            cost: Logarithmic cost function value including regularization.
        """
    
        # calculate cost
        h = self.forward_prop(x_data)[0][-1]
        cost = -1 / x_data.shape[0] * (np.trace(y_data.dot(np.log(h).T)) + np.trace((1 - y_data).dot(np.log(1 - h).T)))

        # add regularization
        reg_cost = 0
        
        for l in range(len(self.w_mats)):
            w_mat = self.w_mats[l]

            reg_w_mat = w_mat[:,1:]
            
            reg_cost += np.trace(reg_w_mat.dot(reg_w_mat.T))

        reg_cost *= lam / 2 / x_data.shape[0]
        
        # total cost
        cost += reg_cost
        
        return cost


    def der_cost_fun(self, y_data, a_mat):
        """ Calculate derivative of cost function w.r.t. hypothesis."""
    
        return (a_mat - y_data) / (a_mat - a_mat**2)


    def grad_descent(self, D_mats, alpha):
        """Perform gradient descent.

        Args:
            D_mats: Derivative of cost w.r.t. weights for each layer.
            alpha: Learning rate.
        """

        for l in range(len(self.w_mats)):
            self.w_mats[l] = self.w_mats[l] - alpha * D_mats[l]


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

        if self.K == 1:
            return np.sum((self.forward_prop(x_data)[0][-1] > 0.5) == y_data)/y_data.shape[0]
        else:       
            return np.sum(np.argmax(self.forward_prop(x_data)[0][-1], axis=1) == np.argmax(y_data, axis=1))/y_data.shape[0]


    def predict(self, x_data):
        """Predict output based on NN weights."""

        if self.K == 1:
            return self.forward_prop(x_data)[0][-1] > 0.5
        else:
            return np.argmax(self.forward_prop(x_data)[0][-1], axis=1)

    
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
        self.conv_data = [self.cost_fun(x_data, y_data, lam)]

        # print iteration
        print_iter = int(0.01 * max_iter)

        for i in range(max_iter):

            if not quiet:
                if i % print_iter == 0 or i == max_iter-1:
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
                D_mats = self.backward_prop(x_batch, y_batch, lam)

                # do gradient descent
                self.grad_descent(D_mats, alpha)

            # check convergence
            self.conv_data.append(self.cost_fun(x_data, y_data, lam))
            if abs((self.conv_data[-2]-self.conv_data[-1])/self.conv_data[-2]) < eps:
                break

            # print information
            if not quiet:            

                if i % print_iter == 0 or i == max_iter-1:
                    stop_time = time.time()
                    print("Iteration: %d - Cost: %f - Time: %f - Alpha: %f" % (i, self.conv_data[-1], stop_time - start_time, alpha))
            
                if i % (10*print_iter) == 0 or i == max_iter-1:
                    print("Accuracy: %.2f %%" % (self.test(x_data, y_data)*100))
            

    