import numpy as np

class Model:

    def __init__(self, train_X, train_Y, test_X, test_Y):
        # initialize training and test data
        self.X_train = train_X
        self.Y_train = train_Y
        self.X_test = test_X
        self.Y_test = test_Y