import numpy as np
from datetime import datetime
from kNN.kNN import kNN
from sklearn.decomposition import PCA

def write_results(data, dist, train=False):
    filename = ""
    if train:
        filename = r'./results/' + dist + "_small_train.txt"
    else: 
        filename = r'./results/' + dist + "_small_test.txt"
    np.savetxt(filename, data, delimiter=",", fmt='%s')	

# train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')[:250]
# test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')[:150]
train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')
test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')
print(f'train shape {train.shape}\ntest shape {test.shape}')

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

distances = ['minkowski']

for dist in distances:
    cls = kNN(X=X_train, y=y_train)

    data_test = [["0/1 loss", "k", "p"]]
    data_train = [["0/1 loss", "k", "p"]]
    for p in range(1, 16):
        # Test data
        y_hat = cls.predict(X_test, dist, LOOCV=False, p=p)
        for i in range(0, 20):
            data_test.append([np.sum(y_hat[:, i] != y_test), i+1, p])

        # Train data
        y_hat = cls.predict(X_train, dist, LOOCV=True)

        for i in range(0, 20):
            data_train.append([np.sum(y_hat[:, i] != y_train), i+1, p])

    write_results(data_test, dist, train=False)
    write_results(data_train, dist, train=True)

