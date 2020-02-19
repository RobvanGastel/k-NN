import numpy as np
import time
from kNN.kNN import kNN

train = np.loadtxt(r'./data/MNIST_train_small.csv', 
    delimiter=',')
test = np.loadtxt(r'./data/MNIST_test_small.csv', 
    delimiter=',')

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

start = time.process_time()
cls = kNN(X=X_train, y=y_train)
y_hat = cls.predict(X_test)
print("processing time: %.2f seconds" % (time.process_time() - start))

for i in range(0, 20):
    print("0/1 loss: ", np.sum(y_hat[:, i] != y_test), "for k: ", i)