import numpy as np

class kNN:

    def __init__(self, X, y):
        # Training data
        self.X = X
        self.y = y
        self.n_classes = len(list(set(self.y)))

    def predict(self, X_input, dist, k_range=range(1, 21)):
        y_hat = np.empty(shape=(X_input.shape[0], max(k_range)))

        for i in range(0, len(X_input)):
            distances = self.distance(X_input[i], dist)

            # Order distance by closest elements
            neighbors = sorted(distances, key=lambda x: x[0])

            for k in k_range:
                k_neighbors = neighbors[:k]
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

    def euclidian(self, x):
        '''Euclidian distance
        '''
        distances = np.sqrt(np.sum(np.power(
            np.subtract(x, self.X), 2), axis=1))
        return np.stack((distances, self.y), axis=-1)


    def manhattan(self, X, Y):
        '''
        Manhattan distance
        https://en.wikipedia.org/wiki/Taxicab_geometry
        '''
        return np.sum(np.abs(X-Y))

    def cosine(self, x):
        distances = np.divide(np.sum(np.multiply(x, self.X), axis=1),
                              np.sqrt(np.multiply(np.sum(np.power(x, 2), axis=1), np.sum(np.power(self.X, 2), axis=1))))
        return np.stack((distances, self.y), axis=-1)

    def mahalanobis(self, X, Y):
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
