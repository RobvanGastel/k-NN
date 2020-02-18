import numpy as np

train = np.loadtxt(r'../data/MNIST_train_small.csv', delimiter=',')
X_train, y_train = train[:,1:], train[:,0]
test = np.loadtxt(r'../data/MNIST_test_small.csv', delimiter=',')
X_test, y_test = test[:,1:], test[:,0]


class kNN:
    def __init__(self):
