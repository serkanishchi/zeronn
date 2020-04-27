import numpy as np
from .NNModel import Model 
from .helpers import sigmoid

class LogisticRegression(Model):

    def __init__(self, train_X, train_Y, test_X, test_Y):
        # initialize training and test data
        super(LogisticRegression, self).__init__(train_X, train_Y, test_X, test_Y)
    
    def initialize_with_zeros(self):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        --

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        
        dim = self.X_train.shape[0]

        w = np.zeros(shape=(dim, 1))
        b = 0

        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))

        return w, b
    
    def propagate(self):
        """
        Implement the cost function and its gradient for the propagation

        Arguments:
        --

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """

        m = self.X_train.shape[1]

        # FORWARD PROPAGATION (FROM X TO COST)
        A = sigmoid(np.dot(self.w.T, self.X_train) + self.b)  # compute activation
        cost = (- 1 / m) * np.sum(self.Y_train * np.log(A) + (1 - self.Y_train) * (np.log(1 - A)))  # compute cost

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = (1 / m) * np.dot(self.X_train, (A - self.Y_train).T)
        db = (1 / m) * np.sum(A - self.Y_train)

        assert(dw.shape == self.w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        grads = {"dw": dw,
                 "db": db}

        return grads, cost
    
    def optimize(self, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        
        costs = []
    
        for i in range(num_iterations):

            # Cost and gradient calculation
            grads, cost = self.propagate()

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule (≈ 2 lines of code)
            self.w = self.w - learning_rate * dw  # need to broadcast
            self.b = self.b - learning_rate * db

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))

        grads = {"dw": dw,
                 "db": db}

        return grads, costs
    
    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        self.w = self.w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = sigmoid(np.dot(self.w.T, X) + self.b)

        for i in range(A.shape[1]):
            # Convert probabilities a[0,i] to actual predictions p[0,i]
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

        assert(Y_prediction.shape == (1, m))

        return Y_prediction
    
    def fit(self, num_iterations=2000, learning_rate=0.5, print_cost=False):
        """
        Fit the logistic regression model

        Arguments:
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """
        
        # initialize parameters with zeros
        self.w, self.b = self.initialize_with_zeros()
        
        # Gradient descent (≈ 1 line of code)
        grads, costs = self.optimize(num_iterations, learning_rate, print_cost)

        # Predict test/train set examples (≈ 2 lines of code)
        Y_prediction_test = self.predict(self.X_test)
        Y_prediction_train = self.predict(self.X_train)
        
        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - self.Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - self.Y_test)) * 100))


        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test, 
             "Y_prediction_train" : Y_prediction_train, 
             "w" : self.w, 
             "b" : self.b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}

        return d