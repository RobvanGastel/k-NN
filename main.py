import numpy as np
from datetime import datetime
from kNN.kNN import kNN

def write_results(data, dist, small, train, LOOCV):
    filename = dist
    if small:
        filename += '_small'
    filename += '_train' if train else '_test'
    if LOOCV:
        filename += '_LOOCV'

    np.savetxt(f'./results/{filename}.txt', data, delimiter=",", fmt='%s')	

# train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')
# test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')[:150]
train = np.loadtxt(r'./data/MNIST_train_small.csv', delimiter=',')#[:150]
test = np.loadtxt(r'./data/MNIST_test_small.csv', delimiter=',')
# print(f'train shape {train.shape}\ntest shape {test.shape}')

X_train, y_train = train[:,1:], train[:,0]
X_test, y_test = test[:,1:], test[:,0]

info = [['0/1 loss', 'k', 'p']]

cls = kNN(X=X_train, y=y_train)
for p in range(1, 11):
    y_hat = cls.predict(X_train, 'minkowski', LOOCV=True, p=p)

    for k in range(1, 21):
        info.append([np.sum(y_hat[:, k-1] != y_train), k, p])

write_results(info, 'minkowski', small=True, train=True, LOOCV=True)


# train_set = ['./data/train_2.csv', './data/train_3.csv', './data/train_4.csv', './data/train_5.csv', 
#              './data/train_6.csv', './data/train_7.csv', './data/train_8.csv', 
#              './data/train_9.csv', './data/train_10.csv']
# dist = "euclidian"
 
# for t_set in train_set:
#     train = np.loadtxt(t_set, delimiter=',')
#     cls = kNN(X=train, y=y_train)

#     data_test = [["0/1 loss", "k"]]
#     data_train = [["0/1 loss", "k"]]
#     # Test data
#     y_hat = cls.predict(X_test, dist, LOOCV=False)
#     for i in range(0, 20):
#         data_test.append([np.sum(y_hat[:, i] != y_test), i+1])

#     # Train data
#     y_hat = cls.predict(X_train, dist, LOOCV=False)
#     for i in range(0, 20):
#         data_train.append([np.sum(y_hat[:, i] != y_train), i+1])

#     index = i+2
#     name = f"PCA_euclidian_{index}"
#     write_results(data_test, name, train=False)
#     write_results(data_train, name, train=True)

