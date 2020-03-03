import numpy as np
import time
from kNN.kNN import kNN
start = time.process_time()

train = np.loadtxt(r'./data/MNIST_train_small.csv', 
    delimiter=',')[:250]
test = np.loadtxt(r'./data/MNIST_test_small.csv', 
    delimiter=',')[:50]

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

cls = kNN(X=X_train, y=y_train)
y_hat = cls.predict(X_test, 'euclidian')

for i in range(0, 20):
    print("0/1 loss: ", np.sum(y_hat[:, i] != y_test), "for k: ", i+1)

# def cross_validate(X, y, model):
#     loss = []
#     X_k = X
#     y_k = y

#     y_hat = []
#     for i in range(0, len(X)):
#         X_k = X[:i] + X[i+1:]
#         y_k = y[:i] + y[i+1:]
#         y_hat = model.predict(X_k)
#         # Default 0/1 loss
#         loss.append(np.sum(y_hat[:, i] != y_k))

#         X_k = []
#         y_k = []
#     return loss

# cross_validate(X_test, y_test, cls)

print("processing time: %.2f seconds" % (time.process_time() - start))