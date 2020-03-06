import numpy as np
from datetime import datetime
from kNN.kNN import kNN
import os

p = 8
info = np.loadtxt(r'results/QE/minkowski_train_LOOCV.txt', delimiter=',')
info = np.atleast_2d(info)

def write_results(data, dist, small, train, LOOCV):
    filename = dist
    if small:
        filename += '_small'
    filename += '_train' if train else '_test'
    if LOOCV:
        filename += '_LOOCV'

    np.savetxt(f'./results/QE/{filename}.txt', data, delimiter=",", fmt='%s')	

train = np.loadtxt(r'./data/MNIST_train.csv', delimiter=',')
# train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')[:500]

X_train, y_train = train[:,1:], train[:,0]

cls = kNN(X=X_train, y=y_train)

while True:
    if info.size == 0:
        last_index = int(0)
    else:
        last_index = int(info[-1][0])
    if last_index == int(len(X_train)):
        print('DONE!')
        break
    print('last_index', last_index, '/', len(X_train))

    y_hat = cls.predict(
        X_train[last_index+1:last_index+101, :],
        'minkowski',
        LOOCV=True,
        p=p
    )

    for k in range(1, 21):
        info = np.vstack((
            info,
            np.array([
                last_index+100,
                np.sum(y_hat[:, k-1] != y_train[last_index+1:last_index+101]),
                k,
                p
            ])
        ))

    write_results(info, 'minkowski', small=False, train=True, LOOCV=True)