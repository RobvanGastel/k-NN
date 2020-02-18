import numpy as np

def euclidian(X, Y):
    return np.sqrt(np.sum(np.power(np.subtract(X, Y), 2)))

def cosine(X, Y):
    pass

def mahalonobis(X, Y):
    pass