import numpy as np
from kNN.kNN import kNN

train = np.loadtxt(r'./data/MNIST_train_small.csv', 
    delimiter=',')
test = np.loadtxt(r'./data/MNIST_test_small.csv', 
    delimiter=',')

X_train, y_train = train[:,1:], train[:,0]


# TODO: Remove subset the train
X_train, y_train = X_train[:100],  y_train[:100]


X_test, y_test = test[:,1:], test[:,0]

cls = kNN(X=X_train, y=y_train)
cls.predict(X_test[1], y_test[1])
