import numpy as np
from datetime import datetime
from kNN.kNN import kNN

def write_results(y_hat, y, dist, train=False):
    filename = ""
    now = datetime.now()
    if train:
        filename = r'./results/' + now.strftime("%m/%d/%Y, %H:%M:%S") + dist + "_small_train.txt"
    else: 
        filename = r'./results/' + dist + "_small_test.txt"

    data = [["0/1 loss", "k"]]
    for i in range(0, 20):
        data.append([np.sum(y_hat[:, i] != y), i+1])
        print("0/1 loss: ", np.sum(y_hat[:, i] != y), "for k: ", i+1)
    np.savetxt(filename, data, delimiter=",", fmt='%s')	

# train = np.loadtxt(r'./data/MNIST_train.csv', delimiter=',')[:250]
# test = np.loadtxt(r'./data/MNIST_test.csv', delimiter=',')[:150]
train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')
test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')
print(f'train shape {train.shape}\ntest shape {test.shape}')

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

distances = ['manhattan', 'cosine', 'euclidian']

for dist in distances:
    cls = kNN(X=X_train, y=y_train)

    # Test data
    y_hat = cls.predict(X_test, dist, LOOCV=False)
    write_results(y_hat, y_test, dist, train=False)

    # Train data
    y_hat = cls.predict(X_train, dist, LOOCV=True)
    write_results(y_hat, y_train, dist, train=True)