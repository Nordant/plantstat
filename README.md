# PlantStat

A package with a set of functions for fast and convenient statistical processing of experimental data. Also includes simple AutoML algorithms for classification and regression. Designed for needs of the LPBPS (Laboratory of Physiology and Biochemistry of Plant Stress; Kharkiv, Ukraine).

The package is based on such basic packages as:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [scipy](https://docs.scipy.org/doc/scipy/reference/index.html)
- [statsmodels](https://www.statsmodels.org/stable/index.html)

## Installing from the source:
`pip install git+https://github.com/Nordant/plantstat.git#egg=plantstat`

## PlantStat package on Kaggle:
[Package](https://www.kaggle.com/maksymshkliarevskyi/plantstat-package-statistics-and-automl)

[Notebook (PlantStat package: quick overview of possibilities)](https://www.kaggle.com/maksymshkliarevskyi/plantstat-package-quick-overview-of-possibilities)

## AutoML algorithms:
| AutoML_Classifier | AutoML_Regressor | Clusterer |
| --- | --- | --- |
| LogisticRegression | LinearRegression | KMeans |
| LinearSVC | Ridge | DBSCAN |
| KNN | Lasso | |
| DecisionTree | ElasticNet | |
| RandomForest | RandomForest | |
| GradientBoosting | GradientBoosting | |
| XGB | XGB | |
|  | SVR | |

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
a.basic_stats(save = True)

# Statistical tests
a.var_compare(save = True)
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

- Clusterer - the main class for regression.
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
