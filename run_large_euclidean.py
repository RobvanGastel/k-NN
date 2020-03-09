import numpy as np
from datetime import datetime
from kNN.kNN import kNN

def write_results(data, dist, train=False):
    filename = ""
    if train:
        filename = r'./results/' + dist + "_large_train_LOOCV.txt"
    else: 
        filename = r'./results/' + dist + "_large_test_LOOCV.txt"
    np.savetxt(filename, data, delimiter=",", fmt='%s')	

train = np.loadtxt(r'./data/MNIST_train.csv', delimiter=',')
test = np.loadtxt(r'./data/MNIST_test.csv', delimiter=',')
print(f'train shape {train.shape}\ntest shape {test.shape}')

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

distances = ['euclidean']

for dist in distances:
    cls = kNN(X=X_train, y=y_train)
    data_test = [["0/1 loss", "k"]]
    data_train = [["0/1 loss", "k"]]

    # Test data
    y_hat = cls.predict(X_test, dist, LOOCV=False)
    for i in range(0, 20):
        data_test.append([np.sum(y_hat[:, i] != y_test), i+1])

    # Train data
    y_hat = cls.predict(X_train, dist, LOOCV=True)
    for i in range(0, 20):
        data_train.append([np.sum(y_hat[:, i] != y_train), i+1])

    write_results(data_test, dist, train=False)
    write_results(data_train, dist, train=True)

