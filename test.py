'''
An example of using the package.
'''
import numpy as np
from scipy.stats import *
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

import plantstat
print(plantstat.__description__)
print(plantstat.__version__)
print(plantstat.__author__)

################################################################
### -------------------------------------------------------- ###
### -------------------- Variable_Analyzer ----------------- ###
### -------------------------------------------------------- ###
################################################################
from plantstat import Variable_Analyzer

# Data example
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

# Return data labels (variables' names)
a.labels

# Return outliers and data without outliers
a.outliers()
a.cleaned_data

# Return boxplots for all variables (with or without saving in local directory)
a.boxplot()
a.boxplot(save = True)

# Variables' lengths
a.var_len()

# Data Frame with basic statistics (with or without saving in local directory)
a.basic_stats()
a.basic_stats(save = True)

# Data Frame with statistics tests (with or without saving in local directory)
a.var_compare()
a.var_compare(save = True)

# Pairs of variables
a.get_pairs()
a.get_pairs(indices = True)

# Correlation matrix (with and without heatmap) (with or without saving in local directory)
a.corrs()
a.corrs(method = 'pearson', heatmap = True, save = True)

# QQplots for all variables (with or without saving in local directory)
a.QQplot()
a.QQplot(save = True)

# Pairplot (with or without saving in local directory)
a.pair_plot()
a.pair_plot(save = True)


################################################################
### -------------------------------------------------------- ###
### ------------------------- Auto_ML ---------------------- ###
### -------------------------------------------------------- ###
################################################################
### -------------------------------------------------------- ###
### -------------------- AutoML_Classifier ----------------- ###
### -------------------------------------------------------- ###
################################################################
from plantstat import AutoML_Classifier

# Data example
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
y = iris.target
class_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Model
model = AutoML_Classifier(n_iter = 100)
model.fit(X_train, y_train)

# Model detailed information

# model.cv_results_
# model.best_estimator_
# model.best_pipeline

# Prediction and classification report (with prediction saving)
model.predict(X_test, save = True)
model.predict_proba(X_test, save = True, f_format = 'csv')[:5]
model.classification_report(X_test, y_test, labels = class_names, cmap = 'cividis', save = True)

# AutoML model without some algorithms
model = AutoML_Classifier(n_iter = 100, XGB = False, GradientBoosting = False)
model.fit(X_train, y_train)
model.classification_report(X_test, y_test, labels = class_names)


################################################################
### -------------------------------------------------------- ###
### --------------------- AutoML_Regressor ----------------- ###
### -------------------------------------------------------- ###
################################################################
from plantstat import AutoML_Regressor

# Data example
X, y = datasets.load_boston(return_X_y = True)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Model
model = AutoML_Regressor(n_iter = 100)
model.fit(X_train, y_train)

# Model detailed information

# model.cv_results_
# model.best_estimator_
# model.best_pipeline

# Predcition and report (with prediction saving)
model.predict(X_test, save = True)[:10]
model.prediction_report(X_test, y_test, save = True)

# AutoML model without some algorithms
model = AutoML_Regressor(n_iter = 100, XGB = False, GradientBoosting = False)
model.fit(X_train, y_train)
model.prediction_report(X_test, y_test)
