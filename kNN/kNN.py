import numpy as np
from tqdm import tqdm
from time import process_time

class kNN:

    def __init__(self, X, y):
        # Training data
        self.X = X
        self.y = y
        self.n_classes = len(list(set(self.y)))

    def predict(self, X_input, dist, k_range=range(1, 21), LOOCV=False, **kwargs):
        print(f'start predict {dist}')
        start_predict = process_time()
        y_hat = np.empty(shape=(X_input.shape[0], max(k_range)))

        for i, x_input in enumerate(tqdm(X_input)):
            distances = self.get_distance(x_input, dist, kwargs)
            # Order distance by closest elements
            neighbors = sorted(distances, key=lambda x: x[0])

            for k in k_range:
                k_neighbors = neighbors[int(LOOCV) : k + int(LOOCV)]
                y_hat[i][k-1] = self.__majority_vote(k_neighbors)

        print("predict processing time: %.2f seconds" % (process_time() - start_predict))
        return y_hat

    def get_distance(self, x, dist, kwargs):
        # TODO: Add more distance measures
        if dist == "euclidean":
            return self.euclidean(x)
        if dist == "minkowski":
            if 'p' in kwargs:
                return self.minkowski(x, p=kwargs['p'])
            raise "input 'p' value"
        if dist == "cosine":
            return self.cosine(x)
        if dist == 'manhattan':
            return self.manhattan(x)

    def __majority_vote(self, neighbors):
        '''Majority vote for the k nearest neighbors chosen
        '''
        votes = [0] * self.n_classes
        for n in neighbors:
            votes[int(n[1])] += 1
        
        # Indices which have the same vote
        indices = np.argwhere(votes == np.amax(votes)).flatten().tolist()

        for n in neighbors:
            if int(n[1]) in indices:
                return n[1]

    def euclidean(self, x):
        '''Euclidean distance
        '''
        distances = np.sqrt(np.sum(np.power(
            np.subtract(x, self.X), 2), axis=1))
        return np.stack((distances, self.y), axis=-1)


    def manhattan(self, x):
        '''Manhattan distance
        '''
        q1 = x-self.X
        q2 = np.abs(q1)
        distances = np.sum(q2, axis=1)
        return np.stack((distances, self.y), axis=-1)

    def cosine(self, x):
        '''Cosine similarity
        '''
        distances = []
        for _, x_i in enumerate(self.X):
            distances.append(1- (np.dot(x, x_i) / np.linalg.norm(x_i) * np.linalg.norm(x)))
        return np.stack((distances, self.y), axis=-1)
    
    def minkowski(self, x, p):
        distances = np.power(np.sum(np.power(
            np.abs(np.subtract(x, self.X)), p), axis=1), 1/p)

        return np.stack((distances, self.y), axis=-1)
