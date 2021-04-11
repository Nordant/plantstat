"""
Created on Thu Apr 5 2021
@author: Nordant
"""

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

### ---------------------------------------------- ###
### --------------- Variable_Analyzer ------------ ###
### ---------------------------------------------- ###

import numpy as np
from scipy.stats import *

from plantstat import Variable_Analyzer

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
a.labels

a.outliers()
a.cleaned_data


a.boxplot()

a.var_len()

a.basic_stats()

a.var_compare()

a.get_pairs()
a.get_pairs(indices = True)

a.corrs()
a.corrs(method = 'pearson', heatmap = True)

a.QQplot()

a.pair_plot()




### ---------------------------------------------- ###
### -------------------- Auto_ML ----------------- ###
### ---------------------------------------------- ###

# AutoML_Classifier
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from plantstat import AutoML_Classifier

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 0)

model = AutoML_Classifier(n_iter = 100)
model.fit(X_train, y_train)

# model.cv_results_
# model.best_estimator_
# model.best_pipeline

model.predict(X_test)
model.predict_proba(X_test)[:5]

model.classification_report(X_test, y_test, labels = class_names, cmap = 'cividis')




# AutoML_Regressor
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from plantstat import AutoML_Regressor

X, y = datasets.load_boston(return_X_y = True)
X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 0)

model = AutoML_Regressor(n_iter = 100)
model.fit(X_train, y_train)

# model.cv_results_
# model.best_estimator_
# model.best_pipeline

model.predict(X_test)[:10]

model.prediction_report(X_test, y_test)
