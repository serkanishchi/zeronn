import numpy as np
from .Optimizer import Optimizer

class AdamOptimizer(Optimizer):

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, t=2):
        """
        v -- python dictionary containing the current velocity:
                v['dW' + str(l)] = ...
                v['db' + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
        beta -- the momentum hyperparameter, scalar
        t -- 
        """
        super(AdamOptimizer, self).__init__()
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = t
        self.v = {}
        self.s = {}
    
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
        parameters = super(AdamOptimizer, self).initialize(layer_dims, initialization="he")

        L = len(parameters) // 2 # number of layers in the neural networks

        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            self.v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
            self.v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

            self.s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l + 1)])
            self.s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l + 1)])
        
        return parameters
    
    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using Adam

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
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary
        self.t += 1

        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            self.v["dW" + str(l + 1)] = self.beta1 * self.v["dW" + str(l + 1)] + (1 - self.beta1) * grads['dW' + str(l + 1)]
            self.v["db" + str(l + 1)] = self.beta1 * self.v["db" + str(l + 1)] + (1 - self.beta1) * grads['db' + str(l + 1)]

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l + 1)] = self.v["dW" + str(l + 1)] / (1 - np.power(self.beta1, self.t))
            v_corrected["db" + str(l + 1)] = self.v["db" + str(l + 1)] / (1 - np.power(self.beta1, self.t))

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            self.s["dW" + str(l + 1)] = self.beta2 * self.s["dW" + str(l + 1)] + (1 - self.beta2) * np.power(grads['dW' + str(l + 1)], 2)
            self.s["db" + str(l + 1)] = self.beta2 * self.s["db" + str(l + 1)] + (1 - self.beta2) * np.power(grads['db' + str(l + 1)], 2)

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l + 1)] = self.s["dW" + str(l + 1)] / (1 - np.power(self.beta2, self.t))
            s_corrected["db" + str(l + 1)] = self.s["db" + str(l + 1)] / (1 - np.power(self.beta2, self.t))

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(self.s["dW" + str(l + 1)] + self.epsilon)
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(self.s["db" + str(l + 1)] + self.epsilon)

        return parameters
    
    def __eq__(self, compare): 
        if(compare == "adam"): 
            return True
        else:
            return False