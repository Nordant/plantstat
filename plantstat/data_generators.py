import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_blobs, make_regression

class ClusterData:
    '''
    Generates data for clustering with random number of clusters (in a given range).
    Args:
        n_samples - the number of samples (default = 1000)
        n_features - the number of features (default = 2)
        cluster_std - a cluster standard deviation (default = 1.0)
        centers_range - a range of random centroids number (default = (2, 10))
        random_state - a random state parameter for single 'centers_range' value
        return_labels - return label data or not (default = False)
    '''
    def __init__(self, n_samples=1000, n_features=2, cluster_std=1.0,
                 centers_range=(2, 10), random_state=None, return_labels=False):
        self.n_samples = n_samples
        self.n_features = n_features
        self.cluster_std = cluster_std
        self.centers_range = centers_range
        self.random_state = random_state
        self.return_labels = return_labels
    
    def generate(self, save=False, f_format='excel'):
        '''
        Args:
            save - save prediction in local directory or not
            f_format - format of data saving (if save = True): 'csv' or 'excel' (default)
        '''
        assert f_format in {'excel', 'csv'}
        centers = random.sample(range(self.centers_range[0], self.centers_range[1]), 1)[0]
        self.features, self.labels = make_blobs(n_samples = self.n_samples,
                                                n_features = self.n_features,
                                                centers = centers, 
                                                cluster_std = self.cluster_std, 
                                                random_state = self.random_state)

        if save == True and f_format == 'excel':
            writer = pd.ExcelWriter('ClusterData.xlsx', engine='xlsxwriter')
            pd.DataFrame(self.features).to_excel(writer, sheet_name='data')
            if self.return_labels == True:
                pd.DataFrame(self.labels).to_excel(writer, sheet_name='labels')
            writer.save()
        elif save == True and f_format == 'csv':
            pd.DataFrame(self.features).to_csv('ClusterData_features.csv')
            if self.return_labels == True:
                pd.DataFrame(self.labels).to_csv('ClusterData_labels.csv')
        else:
            pass

        if self.return_labels == False:
            return self.features
        elif self.return_labels == True:
            return self.features, self.labels
        else:
            raise ValueError('Please, set `return_labels` True or False')


class RegressionData:
    '''
    Generates data for regression.
    Args:
        n_samples - the number of samples (default = 1000)
        n_features - the number of features (default = 2)
        n_informative - the number of informative features (default = 2)
        n_targets - the number of regression targets (default = 1)
        bias - the bias term in the underlying linear model (default = 0.0)
        noise - the standard deviation of the gaussian noise applied to the output (default = 0.0)
        shuffle - shuffle the samples and the features (default = True)
        random_state - a random state parameter for single 'centers_range' value
        return_labels - return label data or not (default = False)
    '''

    def __init__(self, n_samples=1000, n_features=2, n_informative=2, n_targets=1,
                 bias=0.0, noise=0.0, shuffle=True, random_state=None, return_labels=True):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_targets = n_targets
        self.bias = bias
        self.noise = noise
        self.shuffle = shuffle
        self.random_state = random_state
        self.return_labels = return_labels

    def generate(self, save=False, f_format='excel'):
        '''
        Args:
            save - save prediction in local directory or not
            f_format - format of data saving (if save = True): 'csv' or 'excel' (default)
        '''
        assert f_format in {'excel', 'csv'}
        self.features, self.labels = make_regression(n_samples=self.n_samples,
                                                     n_features=self.n_features,
                                                     n_informative = self.n_informative,
                                                     n_targets = self.n_targets,
                                                     bias = self.bias,
                                                     noise = self.noise,
                                                     shuffle = self.shuffle,
                                                     random_state=self.random_state)

        if save == True and f_format == 'excel':
            writer = pd.ExcelWriter('RegressionData.xlsx', engine='xlsxwriter')
            pd.DataFrame(self.features).to_excel(writer, sheet_name='data')
            if self.return_labels == True:
                pd.DataFrame(self.labels).to_excel(writer, sheet_name='labels')
            writer.save()
        elif save == True and f_format == 'csv':
            pd.DataFrame(self.features).to_csv('RegressionData_features.csv')
            if self.return_labels == True:
                pd.DataFrame(self.labels).to_csv('RegressionData_labels.csv')
        else:
            pass

        if self.return_labels == False:
            return self.features
        elif self.return_labels == True:
            return self.features, self.labels
        else:
            raise ValueError('Please, set `return_labels` True or False')