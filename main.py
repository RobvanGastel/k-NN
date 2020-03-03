import numpy as np
from kNN.kNN import kNN

# train = np.loadtxt(r'./data/MNIST_train.csv', delimiter=',')#[:250]
# test = np.loadtxt(r'./data/MNIST_test.csv', delimiter=',')#[:150]
train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')
test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')
print(f'train shape {train.shape}\ntest shape {test.shape}')

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

cls = kNN(X=X_train, y=y_train)
y_hat = cls.predict(X_train, 'minkowski', LOOCV=False, p=5) # minkowski: param int p

for i in range(0, 20):
    print("0/1 loss: ", np.sum(y_hat[:, i] != y_train), "for k: ", i+1)