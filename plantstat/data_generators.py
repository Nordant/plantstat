import numpy as np
import random
from sklearn.datasets import make_blobs

class ClusterData:
    '''
    Generates data for clustering with random number of clusters (in a given range).
    Args:
        n_samples - the number of samples (default = 1000)
        n_features - the number of features (default = 2)
        cluster_std - a cluster standard deviation (default = 1.0)
        centers_range - a range of random centroids number (default = (2, 10))
        random_state - a random state parameter for single 'centers_range' value
    '''
    def __init__(self, n_samples = 1000, n_features = 2, cluster_std = 1.0, 
                 centers_range = (2, 10), random_state = None):
        self.n_samples = n_samples
        self.n_features = n_features
        self.cluster_std = cluster_std
        self.centers_range = centers_range
        self.random_state = random_state
    
    def generate(self):
        centers = random.sample(range(self.centers_range[0], self.centers_range[1]), 1)[0]
        self.features, self.labels = make_blobs(n_samples = self.n_samples,
                                                n_features = self.n_features,
                                                centers = centers, 
                                                cluster_std = self.cluster_std, 
                                                random_state = self.random_state)
        return self.features