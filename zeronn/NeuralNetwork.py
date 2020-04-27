import numpy as np
from .Model import Model 
from .helpers import sigmoid

class NeuralNetwork(Model):

    def __init__(self, train_X, train_Y, test_X, test_Y):
        # initialize training and test data
        super(NeuralNetwork, self).__init__(train_X, train_Y, test_X, test_Y)
        
        # initialize layer sizes of input, hidden and output
        # n_input -- the size of the input layer
        # n_hidden -- the size of the hidden layer
        # n_output -- the size of the output layer
        self.n_input, self.n_hidden, self.n_output = self.layer_sizes()
    
    def layer_sizes(self):
        """
        Arguments:
        --

        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
        n_x = self.X_train.shape[0] # size of input layer
        n_h = 4
        n_y = self.Y_train.shape[0] # size of output layer
        return (n_x, n_h, n_y)
    
    def initialize_parameters(self):
        """
        Argument:
        --

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.

        W1 = np.random.randn(self.n_hidden, self.n_input) * 0.01
        b1 = np.zeros(shape=(self.n_hidden, 1))
        W2 = np.random.randn(self.n_output, self.n_hidden) * 0.01
        b2 = np.zeros(shape=(self.n_output, 1))

        assert (W1.shape == (self.n_hidden, self.n_input))
        assert (b1.shape == (self.n_hidden, 1))
        assert (W2.shape == (self.n_output, self.n_hidden))
        assert (b2.shape == (self.n_output, 1))

        return W1, b1, W2, b2
    
    def forward_propagation(self, X):
        """
        Argument:
        --

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """

        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2, cA2 = sigmoid(Z2)

        assert(A2.shape == (1, X.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}

        return A2, cache
    
    def compute_cost(self, A2):
        """
        Computes the cross-entropy cost given in equation (13)

        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost given equation (13)
        """

        m = self.Y_train.shape[1] # number of example

        # Compute the cross-entropy cost
        logprobs = np.multiply(np.log(A2), self.Y_train) + np.multiply((1 - self.Y_train), np.log(1 - A2))
        cost = - np.sum(logprobs) / m

        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
        assert(isinstance(cost, float))

        return cost
    
    def backward_propagation(self, cache):
        """
        Implement the backward propagation using the instructions above.

        Arguments:
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".

        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = self.X_train.shape[1]

        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache['A1']
        A2 = cache['A2']

        # Backward propagation: calculate dW1, db1, dW2, db2. 
        dZ2= A2 - self.Y_train
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(self.W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, self.X_train.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads
    
    def update_parameters(self, grads, learning_rate=1.2):
        """
        Updates parameters using the gradient descent 

        Arguments:
        grads -- python dictionary containing your gradients 

        Returns:
        --
        """
        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        # Update rule for each parameter
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2
        
    def fit(self, n_hidden, learning_rate=1.2, num_iterations=10000, print_cost=False):
        """
        Arguments:
        n_hidden -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations

        Returns:
        --
        """
        
        np.random.seed(3)
        self.n_input, self.n_hidden, self.n_output = self.layer_sizes()
        
        # Initialize parameters
        self.W1, self.b1, self.W2, self.b2 = self.initialize_parameters()
        
        # Loop (gradient descent)

        for i in range(0, num_iterations):

            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = self.forward_propagation(self.X_train)

            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = self.compute_cost(A2)

            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = self.backward_propagation(cache)

            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            self.update_parameters(grads, learning_rate)

            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
                
        # Predict test/train set examples (≈ 2 lines of code)
        Y_prediction_test = self.predict(self.X_test)
        Y_prediction_train = self.predict(self.X_train)
        
        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - self.Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - self.Y_test)) * 100))
        

    def predict(self, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.forward_propagation(X)
        predictions = np.round(A2)

        return predictions