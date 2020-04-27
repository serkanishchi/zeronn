import numpy as np
from .Optimizer import Optimizer

class MomentumOptimizer(Optimizer):

    def __init__(self, beta=0.9):
        """
        v -- python dictionary containing the current velocity:
                v['dW' + str(l)] = ...
                v['db' + str(l)] = ...
        beta -- the momentum hyperparameter, scalar
        """
        super(MomentumOptimizer, self).__init__()
        
        self.beta = beta
        self.v = {}
    
    def initialize(self, layer_dims, initialization="he"):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        initialization -- defines the initialization method ("random" or "he")

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        parameters = super(MomentumOptimizer, self).initialize(layer_dims, initialization="he")

        L = len(parameters) // 2 # number of layers in the neural networks

        # Initialize velocity
        for l in range(L):
            self.v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l+1)])
            self.v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l+1)])
        
        return parameters
    
    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using Momentum

        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        learning_rate -- the learning rate, scalar

        Returns:
        parameters -- python dictionary containing your updated parameters 
        """

        L = len(parameters) // 2 # number of layers in the neural networks

        # Momentum update for each parameter
        for l in range(L):

            # compute velocities
            self.v["dW" + str(l + 1)] = self.beta * self.v["dW" + str(l + 1)] + (1 - self.beta) * grads['dW' + str(l + 1)]
            self.v["db" + str(l + 1)] = self.beta * self.v["db" + str(l + 1)] + (1 - self.beta) * grads['db' + str(l + 1)]
            # update parameters
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * self.v["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * self.v["db" + str(l + 1)]

        return parameters
    
    def __eq__(self, compare): 
        if(compare == "momentum"): 
            return True
        else:
            return False