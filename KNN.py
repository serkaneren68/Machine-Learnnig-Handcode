
import numpy as np
from statistics import mode

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance


def majority_vote(votes):
    return mode(votes)

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def predict(self, x):
        predictions = [self._predict(_x) for _x in x]
        return predictions
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]

        # find the closest k
        closest_k_indices = np.argsort(distances)[0:self.k]
        closest_k_labels = [self.y_train[indice] for indice in closest_k_indices]

        # in case classification case
        most_common = majority_vote(closest_k_labels)
        return most_common
    



    

