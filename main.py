import numpy as np
from datetime import datetime
from kNN.kNN import kNN
from sklearn.decomposition import PCA

def write_results(data, dist, train=False):
    filename = ""
    if train:
        filename = r'./results/' + dist + "_large_train_LOOCV.txt"
    else: 
        filename = r'./results/' + dist + "_large_test_LOOCV.txt"
    np.savetxt(filename, data, delimiter=",", fmt='%s')	

# train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')[:250]
# test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')[:150]
test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')
# print(f'train shape {train.shape}\ntest shape {test.shape}')

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

train_set = ['./data/train_3.csv', './data/train_4.csv', './data/train_5.csv', 
             './data/train_6.csv', './data/train_7.csv', './data/train_8.csv', 
             './data/train_9.csv', './data/train_10.csv']
dist = "euclidian"
 
for t_set in train_set:
    train = np.loadtxt(t_set, delimiter=',')
    cls = kNN(X=t_set, y=y_train)

    data_test = [["0/1 loss", "k", "p"]]
    data_train = [["0/1 loss", "k", "p"]]
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

