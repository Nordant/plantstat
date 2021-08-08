![PlantStat](https://github.com/Nordant/plantstat/blob/main/image.jpeg?raw=true)

# PlantStat

A package with a set of functions for fast and convenient statistical processing of experimental data. Also includes simple AutoML algorithms for classification, regression, clustering, and CV of microscopic images. Developed for needs of the LPBPS (Laboratory of Physiology and Biochemistry of Plant Stress; Kharkiv, Ukraine). The package is used by some educational and scientific institutions in Ukraine.

## Installing from the source:
`pip install git+https://github.com/Nordant/plantstat.git#egg=plantstat`

## PlantStat package on Kaggle:
[Package](https://www.kaggle.com/maksymshkliarevskyi/plantstat-package-statistics-and-automl)

[Notebook (PlantStat package: quick overview of possibilities)](https://www.kaggle.com/maksymshkliarevskyi/plantstat-package-quick-overview-of-possibilities)

## AutoML algorithms:
| AutoML_Classifier | AutoML_Regressor | Clusterer | KNN |
| --- | --- | --- | --- |
| LogisticRegression | LinearRegression | KMeans | NearestNeighbors |
| LinearSVC | Ridge | DBSCAN | |
| KNN | Lasso | | |
| DecisionTree | ElasticNet | | |
| RandomForest | RandomForest | | |
| GradientBoosting | GradientBoosting | | |
| XGB | XGB | | |
|  | SVR | | |

## Data generators:
| Generator |
| --- |
| ClusterData |
| RegressionData |

```python
from plantstat.data_generators import ClusterData
data_gen = ClusterData(n_features = 5, n_samples = 1000, cluster_std = 1.2,
                       centers_range = (4, 5), random_state = 0, return_labels = False)
X = pd.DataFrame(data_gen.generate(save = True))
```

```python
from plantstat.data_generators import RegressionData
data_gen = RegressionData(n_features = 5, n_samples = 1000, n_informative = 3, n_targets = 1,
                          bias = 0.0, noise = 0.2, shuffle = True, random_state = 0, return_labels = True)
X, y = data_gen.generate(save = True)
```

## Examples:
- Variable_Analyzer - the main class for statistical data processing.
```python
from plantstat import Variable_Analyzer
# Define Analyzer
a = Variable_Analyzer(data, labels)

# An example of step-by-step analysis with saving in a local directory
# EDA
a.boxplot(save = True)
a.outliers()
a.corrs(method = 'pearson', heatmap = True, save = True)
a.QQplot(save = True)
a.pair_plot(save = True)

# Basic data statistics
a.basic_stats(p_value = False)
a.basic_stats(p_value = True, save = True)

# Statistical tests
a.var_compare(p_value = False)
a.var_compare(p_value = True, save = True)
```

- AutoML_Classifier - the main class for classification.
```python
from plantstat import AutoML_Classifier
# Define AutoML_Classifier
model = AutoML_Classifier(n_iter = 100)
model.fit(X_train, y_train)

# Model detailed information
# model.cv_results_
# model.best_estimator_
# model.best_pipeline

# Prediction and classification report (with prediction saving)
model.predict(X_test, save = True)
model.predict_proba(X_test, save = True, f_format = 'csv')
model.classification_report(X_test, y_test, labels = class_names, cmap = 'cividis', save = True)

# AutoML model without some algorithms
model = AutoML_Classifier(n_iter = 100, XGB = False, GradientBoosting = False)
model.fit(X_train, y_train)
model.classification_report(X_test, y_test, labels = class_names)
```

- AutoML_Regressor - the main class for regression.
```python
from plantstat import AutoML_Regressor
# Define AutoML_Classifier
model = AutoML_Regressor(n_iter = 100)
model.fit(X_train, y_train)

# Model detailed information
# model.cv_results_
# model.best_estimator_
# model.best_pipeline

# Predcition and report (with prediction saving)
model.predict(X_test, save = True)
model.prediction_report(X_test, y_test, save = True)

# AutoML model without some algorithms
model = AutoML_Regressor(n_iter = 100, XGB = False, GradientBoosting = False)
model.fit(X_train, y_train)
model.prediction_report(X_test, y_test)
```

- Clusterer - the main class for clustering.
```python
from plantstat import Clusterer
from plantstat.data_generators import ClusterData

# Create syntetic data with 5 features, 1000 samples and 4 clusters
data_gen = ClusterData(n_features = 5, n_samples = 1000, cluster_std = 1.2,
                       centers_range = (4, 5), random_state = 0)
X = pd.DataFrame(data_gen.generate())

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
```

- KNN - the main class for Nearest Neighbors similarity finding.
```python
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
```

## OpenStomataPredictor - the main class for stomata open/close classes prediction.
```python
from plantstat.vision.stomata_vision import OpenStomataPredictor

predictor = OpenStomataPredictor('PATH', batch_size = 16)

predictor.predict(save = True)
predictor.visualize(save = True)

predictor.report_

predictor.test_img_paths_
predictor.test_preds_
predictor.test_classes_
```
