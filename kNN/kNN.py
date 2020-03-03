import numpy as np

class kNN:

    def __init__(self, X, y):
        # Training data
        self.X = X
        self.y = y
        self.n_classes = len(list(set(self.y)))
        self.cov_matrix = None

    def predict(self, X_input, dist, k_range=range(1, 21), predict_on_train=False):
        y_hat = np.empty(shape=(X_input.shape[0], max(k_range)))

        for i, x_input in enumerate(X_input):
            distances = self.distance(x_input, dist)
            # Order distance by closest elements
            neighbors = sorted(distances, key=lambda x: x[0])

            for k in k_range:
                k_neighbors = neighbors[int(predict_on_train) : k + int(predict_on_train)]
                y_hat[i][k-1] = self.__majority_vote(k_neighbors)

        return y_hat

    def distance(self, x, dist):
        # TODO: Add more distance measures
        if dist == "euclidian":
            return self.euclidian(x)
        if dist == "minkowski":
            return self.minkowski(x)
        if dist == "cosine":
            return self.cosine(x)
        if dist == 'manhattan':
            return self.manhattan(x)
        if dist == "mahalanobis":
            return self.mahalanobis(x)

    def euclidian(self, x):
        '''Euclidian distance
        '''
        distances = np.sqrt(np.sum(np.power(
            np.subtract(x, self.X), 2), axis=1))
        return np.stack((distances, self.y), axis=-1)


    def manhattan(self, x):
        '''
        Manhattan distance
        https://en.wikipedia.org/wiki/Taxicab_geometry
        '''
        q1 = x-self.X
        q2 = np.abs(q1)
        q3 = np.sum(q2, axis=1)
        distances = q3
        return np.stack((distances, self.y), axis=-1)

    def cosine(self, x):
        '''
        Cosine similarity
        https://en.wikipedia.org/wiki/Cosine_similarity
        '''
        distances = []
        for _, x_i in enumerate(self.X):
            distances.append(1- (np.dot(x, x_i) / np.linalg.norm(x_i) * np.linalg.norm(x)))
        return np.stack((distances, self.y), axis=-1)

    def mahalanobis(self, x):
        '''
        Mahalanobis distance
        https://en.wikipedia.org/wiki/Mahalanobis_distance
        '''
        pass

    
    def minkowski(self, x, p=5):
        distances = np.power(np.sum(np.power(
            np.abs(np.subtract(x, self.X)), p), axis=1), 1/p)
        return np.stack((distances, self.y), axis=-1)
    
    def chebyshev(self, X, Y):
        '''
        Chebyshev distance
        https://iq.opengenus.org/chebyshev-distance/
        '''
        pass

    def chi_square (self, X, Y):
        pass


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
