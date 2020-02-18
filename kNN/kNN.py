import numpy as np
from kNN.distance import euclidian

class kNN:

    def __init__(self, X, y):
        # Training data
        self.X = X
        self.y = y
        self.n_classes = len(list(set(self.y)))

    def evaluate(self, X_test, y_test, k=5, d=euclidian):

        # TODO: Same as predict but iterating over test set
        pass

    def predict(self, X_j, y_j, k=5, dist=euclidian):
        distances = {}

        for i in range(0, len(self.X)):
            di = 0
            for x, y in zip(self.X[i], X_j):
                di += dist(X=x, Y=y)
            distances[i] = (di, int(self.y[i]))

        distances = sorted(distances.items(), key=lambda x: x[1])

        # Look at the k neighbors
        distances = distances[:k]

        # Majority vote on X_j
        return self.__majority_vote(distances)

    def __majority_vote(self, neighbors):
        votes = [0] * self.n_classes
        
        for n in neighbors:
            votes[n[1][1]] += 1
        
        # TODO: Takes first occurence if even numbers
        return votes.index(max(votes))






