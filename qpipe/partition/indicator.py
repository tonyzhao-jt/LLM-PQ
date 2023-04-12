import numpy as np 

class Indicator:
    def __init__(self, layers):
        self.layers = layers

    def random_indicator(self):
        return np.random.rand(self.layers)