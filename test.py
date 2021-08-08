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

# Data Frame with basic statistics (with or without p_values and saving in local directory)
a.basic_stats()
a.basic_stats(p_value = False)
a.basic_stats(save = True)

# Data Frame with statistics tests (with or without p_values and saving in local directory)
a.var_compare()
a.var_compare(p_value = False)
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
X, y = pd.DataFrame(iris.data), iris.target
class_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Model
model = AutoML_Classifier(n_iter = 100)
model.fit(X_train, y_train)

# # Model detailed information
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
from plantstat.data_generators import RegressionData

# Create synthetic data with 5 features (3 informative) and 1000 samples (with plantstat class RegressionData; see data_generators)
data_gen = RegressionData(n_features = 5, n_samples = 1000, n_informative = 3, n_targets = 1,
                          bias = 0.0, noise = 0.2, shuffle = True, random_state = 0, return_labels = True)
X, y = data_gen.generate(save = True)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Model
model = AutoML_Regressor(n_iter = 100)
model.fit(X_train, y_train)

# Model detailed information

# model.cv_results_
# model.best_estimator_
# model.best_pipeline

# Prediction and report (with prediction saving)
model.predict(X_test, save = True)[:10]
model.prediction_report(X_test, y_test, save = True)

# AutoML model without some algorithms
model = AutoML_Regressor(n_iter = 100, XGB = False, GradientBoosting = False)
model.fit(X_train, y_train)
model.prediction_report(X_test, y_test)


################################################################
### -------------------------------------------------------- ###
### ------------------------ Clusterer --------------------- ###
### -------------------------------------------------------- ###
################################################################
from plantstat import Clusterer
from plantstat.data_generators import ClusterData

# Create synthetic data with 5 features, 1000 samples and 4 clusters (with plantstat class ClusterData; see data_generators)
data_gen = ClusterData(n_features = 5, n_samples = 1000, cluster_std = 1.2,
                       centers_range = (4, 5), random_state = 0, return_labels = False)
X = pd.DataFrame(data_gen.generate(save = True))

# Create K-means model for clustering (the model includes PCA with 2 components)
kmeans = Clusterer(is_pca = True, clusterer = 'kmeans')
kmeans.fit(X, save = True)

# Score values for various number of clusters
kmeans.scores

# Preprocessed data (after scaling and PCA)
kmeans.X

# Prediction (basis on fit results)
preds = kmeans.predict(k = 4, save = True)

# The real number of labels
print('Unique labels: %i' %len(np.unique(data_gen.labels)))


# Create DBSCAN model for clustering (the model includes PCA with 2 components)
dbscan = Clusterer(is_pca = True, clusterer = 'DBSCAN')
dbscan.fit(X, save = True)

# Score values for various eps values
dbscan.scores

# Preprocessed data (after scaling and PCA)
dbscan.X

# Prediction (basis on fit results)
preds = dbscan.predict(eps = 0.36, save = True)



################################################################
### -------------------------------------------------------- ###
### --------------------------- KNN ------------------------ ###
### -------------------------------------------------------- ###
################################################################
import pandas as pd
from sklearn.datasets import load_iris
iris = pd.DataFrame(load_iris().data)

from plantstat import KNN

# Create and fit KNN with 5 neighbors
nn = KNN(5)
nn.fit(iris, save = True)

# Find neighbors for data subset
nn.find_neighbors(iris.iloc[:10, :], save = True)

# all kinds of data in the class
# nn.self_distances_
# nn.self_indices_
# nn.model_
# nn.all_data_

# nn.distances
# nn.indices



################################################################
### -------------------------------------------------------- ###
### ------------------------- vision ----------------------- ###
### -------------------------------------------------------- ###
################################################################
### -------------------------------------------------------- ###
### ------------------ OpenStomataPredictor ---------------- ###
### -------------------------------------------------------- ###
################################################################
from plantstat.vision.stomata_vision import OpenStomataPredictor

predictor = OpenStomataPredictor('PATH', batch_size = 16)

predictor.predict(save = True)
predictor.visualize(save = True, n_imgs = 16)

predictor.report_

predictor.test_img_paths_
predictor.test_preds_
predictor.test_classes_
