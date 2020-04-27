import numpy as np

class Optimizer:

    def __init__(self):
        pass
    
    def initialize(self,layer_dims, initialization="he"):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        initialization -- defines the initialization method ("random" or "he")

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        
        np.random.seed(3)
        
        parameters = {}
        L = len(layer_dims)            # number of layers in the network

        # Define the multiplier according to initialization method.
        # Random Initialization uses random weights and scaling factor for weights to break symmetry
        if initialization == "random":
            for l in range(1, L):
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
                parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        # He initialization sqrt(2./layers_dims[l-1]) instead of scaling factor
        elif initialization == "he":
            for l in range(1, L):
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
                parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        return parameters
    
    def update_parameters(self):
        pass