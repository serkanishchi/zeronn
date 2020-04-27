import numpy as np
from .Optimizer import Optimizer

class GDOptimizer(Optimizer):

    def __init__(self):
        super(GDOptimizer, self).__init__()
        
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
        parameters = super(GDOptimizer, self).initialize(layer_dims, initialization="he")
        return parameters
    
    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        learning_rate -- the learning rate, scalar.

        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.

        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        return parameters
    
    def __eq__(self, compare): 
        if(compare == "gd"): 
            return True
        else:
            return False