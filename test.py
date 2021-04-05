"""
Created on Thu Apr 5 2021
@author: Nordant
"""

import numpy as np
from scipy.stats import *

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

from plantstat import Variable_Analyzer

# Example of the data
data = np.array([[54, 27, 43, 30, 29, 23, 71, 68, 64, 66, 64, 70, 49, 49, 
                 55, 45, 48, 49, 38],
                [53, 43, 55, 63, 82, 79, 70, 57, 60, 43, 49, 65, 54],
                [53, 43, 46, 74, 57, 66, 72, 38, 45, 63, 56, 58, 57, 39, 
                 35, 64, 45, 52],
                [36, 45, 23, 83, 70, 82, 77, 41, 37, 48, 55, 52, 22],
                [35, 42, 49, 74, 83, 73, 68, 60, 45, 70, 52, 72, 59, 48, 
                 62, 72, 62, 38]])
labels = ['Control', 'Var1', 'Var2', 'Var3', 'Var4']

# Define Analyzer
a = Variable_Analyzer(data, labels)

# All functions
a.stat(np.mean)
a.stat(iqr)

a.outliers()

a.boxplot()

a.var_len()

a.basic_stats()

a.var_compare()

a.corrs()
a.corrs(method = 'pearson', heatmap = True)

a.QQplot()

a.pair_plot()


