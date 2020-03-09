import numpy as np
from datetime import datetime
from kNN.kNN import kNN

# Write results to file
def write_results(data, dist, small, train, LOOCV):
    filename = dist
    if small:
        filename += '_small'
    filename += '_train' if train else '_test'
    if LOOCV:
        filename += '_LOOCV'
    np.savetxt(f'./results/{filename}.txt', data, delimiter=",", fmt='%s')	

train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')
test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')
print(f'train shape {train.shape}\ntest shape {test.shape}')

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

info = [['0/1 loss', 'k', 'p']]
cls = kNN(X=X_train, y=y_train)

# Checking different parameters
for p in range(1, 15):
    y_hat = cls.predict(X_train, 'minkowski', LOOCV=True, p=p)

    for k in range(1, 21):
        info.append([np.sum(y_hat[:, k-1] != y_train), k, p])

write_results(info, 'minkowski', small=True, train=True, LOOCV=True)