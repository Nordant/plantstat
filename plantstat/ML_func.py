import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from sklearn.metrics import balanced_accuracy_score, classification_report, plot_confusion_matrix
from sklearn.metrics import mean_absolute_error, silhouette_score


class AutoML_Classifier:
    '''
    AutoML algorithm for classification.
    Args:
        scoring_func - parameters of the estimated metric (default = balanced_accuracy) 
        n_iter - the number of iterations of parameters search (default = 50)
        random_state - a random_state parameter (default = 0) 
        cv - a number of cross_validation repeats (default = 5).
        Set False for those algorithms you don't want to use.
    '''
    def __init__(self, scoring_func='balanced_accuracy',
                 n_iter=50, random_state=0, cv=5,
                 LogisticRegression=True, KNN=True,
                 DecisionTree=True, RandomForest=True,
                 LinearSVC=True, GradientBoosting=True,
                 XGB=True):
        self.scoring_func = scoring_func
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        
        self.LogisticRegression = LogisticRegression
        self.KNN = KNN
        self.DecisionTree = DecisionTree
        self.RandomForest = RandomForest
        self.LinearSVC = LinearSVC
        self.GradientBoosting = GradientBoosting
        self.XGB = XGB
        
    def fit(self, X, y):
        '''
        Args:
            X - a data frame with predictors
            y - predicted variable. 
        It selects an optimal machine learning algorithm and performs all 
        the data preprocessing necessary for this algorithm.
        Return:
            best_estimator_
            best_params_ required for prediction,
            detailed cv results.
        '''
        X_train = X
        y_train = y
        
        # All unique cat values
        cat_val = []
        cat_subset = X_train.select_dtypes(include = ['object', 'category', 'bool'])
        for i in cat_subset.columns:
            cat_val.append(list(cat_subset[i].dropna().unique()))
        
        # Preprocessing
        cat_pipeline = Pipeline([('cleaner', SimpleImputer(strategy = 'most_frequent')),
                                ('encoder', OneHotEncoder(sparse = False, categories = cat_val))])
        
        num_pipeline = Pipeline([('cleaner', SimpleImputer()),
                                 ('scaler', StandardScaler())])
        
        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(dtype_exclude = ['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(dtype_include = ['object', 'category', 'bool']))
        ])
        
        # Main pipeline
        model_pipeline_steps = []
        model_pipeline_steps.append(('preprocessor', preprocessor))
        model_pipeline_steps.append(('feature_selector', SelectKBest(f_classif, k = 'all')))
        model_pipeline_steps.append(('estimator', LogisticRegression()))
        model_pipeline = Pipeline(model_pipeline_steps)
        
        
        total_features = preprocessor.fit_transform(X_train).shape[1]
        optimization_grid = []
        
        # ALGORITHMS SELECTION
        # Logistic regression
        if self.LogisticRegression == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [LogisticRegression()]
            })
        
        # K-nearest neighbors
        if self.KNN == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [KNeighborsClassifier()],
                'estimator__weights': ['uniform', 'distance'],
                'estimator__n_neighbors': np.arange(1, 20, 1)
            })
        
        # Decision tree
        if self.DecisionTree == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [DecisionTreeClassifier(random_state = self.random_state)],
                'estimator__criterion': ['gini', 'entropy']
            })
        
        # Random Forest
        if self.RandomForest == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [RandomForestClassifier(random_state = self.random_state)],
                'estimator__n_estimators': np.arange(5, 1000, 20),
                'estimator__criterion': ['gini', 'entropy']
            })
        
        # Linear SVM
        if self.LinearSVC == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean','median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [LinearSVC(random_state = self.random_state)],
                'estimator__C': np.arange(0.1, 1.1, 0.1),
            })

        # Gradient boosting
        if self.GradientBoosting == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [GradientBoostingClassifier(random_state = self.random_state)],
                'estimator__n_estimators': np.arange(5, 1000, 20),
                'estimator__learning_rate': np.linspace(0.01, 1.0, 30),
            })
        
        # XGBoost
        if self.XGB == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [XGBClassifier(random_state = self.random_state)],
                'estimator__n_estimators': np.arange(5, 1000, 20),
                'estimator__learning_rate': np.linspace(0.01, 1.0, 30),
            })
        
        # Search the best estimator
        search = RandomizedSearchCV(
                  model_pipeline,
                  optimization_grid,
                  n_iter = self.n_iter,
                  scoring = self.scoring_func, 
                  n_jobs = -1, 
                  random_state = self.random_state, 
                  verbose = 1,
                  cv = self.cv,
                  return_train_score = True)

        search.fit(X_train, y_train)
        
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_
        self.cv_results_ = search.cv_results_
        
        best_alg = str(self.best_pipeline['estimator']).split('(')[0]
        print('{} was used as the best algorithm!'.format(best_alg))
        
        
    def predict(self, X, save=False, f_format='excel'):
        '''
        Class prediction based on trained AutoML model.
        Args:
            X - a data frame with test data
            save - save prediction in local directory or not
            f_format - format of data saving (if save = True): 'csv' or 'excel' (default)
        Return:
            the numeric classes.
        '''
        assert f_format in {'excel', 'csv'}
        preds = pd.DataFrame(self.best_estimator_.predict(X))
        if save == True and f_format == 'csv':
            preds.to_csv('preds.csv')
        elif save == True and f_format == 'excel':
            preds.to_excel('preds.xlsx', sheet_name = 'preds')
        else:
            pass
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X, save=False, f_format='excel'):
        '''
        Class prediction based on trained AutoML model.
        Args:
            X - a data frame with test data
            save - save prediction in local directory or not
            f_format - format of data saving (if save = True): 'csv' or 'excel' (default)
        Return: 
            the probabilities of classes.
        '''
        assert f_format in {'excel', 'csv'}
        preds = pd.DataFrame(self.best_estimator_.predict_proba(X))
        if save == True and f_format == 'csv':
            preds.to_csv('preds.csv')
        elif save == True and f_format == 'excel':
            preds.to_excel('preds.xlsx', sheet_name = 'preds')
        else:
            pass
        return self.best_estimator_.predict_proba(X)
    
    def classification_report(self, X, y, labels=None, cmap='inferno',
                              save=False):
        '''
        Prediction classification report.
        Args:
            X - a data frame with predictors
            y - predicted variable.
            labels - a list of labels
            cmap - color map
            save - whether to save the output plot in local directory or not
        Return:
            plots
            classification_report
        '''
        report = classification_report(y, self.best_estimator_.predict(X), 
                                       target_names = labels)
        plot_confusion_matrix(self.best_estimator_, X, y,
                              display_labels = labels, cmap = cmap)
        if save == True:
            plt.savefig('Preds_Heatmap.png', dpi = 200)
        plt.show()
        return print(report)
    
    
class AutoML_Regressor:
    '''
    AutoML algorithm for regression.
    Args:
        scoring_func - parameters of the estimated metric (default = neg_mean_squared_error) 
        n_iter - the number of iterations of parameters search (default = 50)
        random_state - a random_state parameter (default = 0) 
        cv - a number of cross_validation repeats (default = 5).
        Set False for those algorithms you don't want to use.
    '''
    def __init__(self, scoring_func='neg_mean_squared_error',
                 n_iter=50, random_state=0, cv=5,
                 LinearRegression=True, Lasso=True,
                 Ridge=True, ElasticNet=True,
                 RandomForest=True, SVR=True,
                 GradientBoosting=True, XGB=True):
        self.scoring_func = scoring_func
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        
        self.LinearRegression = LinearRegression
        self.Lasso = Lasso
        self.Ridge = Ridge
        self.ElasticNet = ElasticNet
        self.SVR = SVR
        self.RandomForest = RandomForest
        self.GradientBoosting = GradientBoosting
        self.XGB = XGB
        
    def fit(self, X, y):
        '''
        Args:
            X - a data frame with predictors
            y - predicted variable. 
        It selects an optimal machine learning algorithm and performs all 
        the data preprocessing necessary for this algorithm.
        Return:
            best_estimator_
            best_params_ required for prediction,
            detailed cv results.
        '''
        X_train = X
        y_train = y
        
        # All unique cat values
        cat_val = []
        cat_subset = X_train.select_dtypes(include = ['object', 'category', 'bool'])
        for i in cat_subset.columns:
            cat_val.append(list(cat_subset[i].dropna().unique()))
        
        if len(cat_val) > 0:
            print('The data has categorical predictors: {}'.format(cat_subset.columns))
        
        # Preprocessing
        cat_pipeline = Pipeline([('cleaner', SimpleImputer(strategy = 'most_frequent')),
                                ('encoder', OneHotEncoder(sparse = False, categories = cat_val))])
        
        num_pipeline = Pipeline([('cleaner', SimpleImputer()),
                                 ('scaler', StandardScaler())])
        
        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(dtype_exclude = ['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(dtype_include = ['object', 'category', 'bool']))
        ])
        
        # Main pipeline
        model_pipeline_steps = []
        model_pipeline_steps.append(('preprocessor', preprocessor))
        model_pipeline_steps.append(('feature_selector', SelectKBest(f_classif, k = 'all')))
        model_pipeline_steps.append(('estimator', LinearRegression()))
        model_pipeline = Pipeline(model_pipeline_steps)
        
        
        total_features = preprocessor.fit_transform(X_train).shape[1]
        optimization_grid = []
        
        # ALGORITHMS SELECTION
        # Linear Regression
        if self.LinearRegression == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [LinearRegression()],
            })
        
        # Lasso (L1)
        if self.Lasso == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [Lasso(random_state = self.random_state)],
                'estimator__alpha': np.arange(0.001, 1.01, 0.05),
            })

        # Ridge (L2)
        if self.Ridge == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [Ridge(random_state = self.random_state)],
                'estimator__alpha': np.arange(0.001, 1.01, 0.05),
            })
        
        # ElasticNet (L1+L2)
        if self.ElasticNet == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [ElasticNet(random_state = self.random_state)],
                'estimator__alpha': np.arange(0.001, 1.01, 0.05),
                'estimator__l1_ratio': np.arange(0.0, 1.01, 0.2),
            })
        
        # SVR
        if self.SVR == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean','median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [SVR()],
                'estimator__C': np.concatenate([np.arange(0.1, 1.1, 0.1), 
                                                np.arange(10, 101, 10)]),
                'estimator__epsilon': np.array([0.01, 0.1, 1]),
            })
        
        # Random Forest Regressor
        if self.RandomForest == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [RandomForestRegressor(random_state = self.random_state)],
                'estimator__n_estimators': np.arange(5, 1000, 20),
                'estimator__criterion': ['gini', 'entropy'],
                'estimator__max_features': ['auto', 'sqrt', 'log2'],
            })

        # Gradient boosting
        if self.GradientBoosting == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [GradientBoostingRegressor(random_state = self.random_state)],
                'estimator__n_estimators': np.arange(5, 1000, 20),
                'estimator__learning_rate': np.linspace(0.01, 1.0, 30),
                'estimator__subsample': np.arange(0.5, 1.01, 0.1),
                'estimator__max_depth': np.arange(2, 11, 1),
                'estimator__max_features': ['auto', 'sqrt', 'log2'],
            })

        # XGBoost
        if self.XGB == True:
            optimization_grid.append({
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [XGBRegressor(random_state = self.random_state)],
                'estimator__n_estimators': np.arange(50, 1001, 50),
                'estimator__learning_rate': np.linspace(0.01, 1, 100),
                'estimator__subsample': np.arange(0.5, 1.01, 0.1),
                'estimator__max_depth': np.arange(2, 11, 1),
                'estimator__colsample_bytree': np.arange(0.4, 1.01, 0.1),
                'estimator__reg_alpha': np.arange(0, 1.01, 0.1),
                'estimator__reg_lambda': np.arange(0, 1.01, 0.1),
            })
        
        # Search the best estimator
        search = RandomizedSearchCV(
                  model_pipeline,
                  optimization_grid,
                  n_iter = self.n_iter,
                  scoring = self.scoring_func, 
                  n_jobs = -1, 
                  random_state = self.random_state, 
                  verbose = 1,
                  cv = self.cv,
                  return_train_score = True)

        search.fit(X_train, y_train)
        
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_
        self.cv_results_ = search.cv_results_
        
        best_alg = str(self.best_pipeline['estimator']).split('(')[0]
        print('{} was used as the best algorithm!'.format(best_alg))
        
        
    def predict(self, X, save=False, f_format='excel'):
        '''
        Prediction based on trained AutoML model.
        Args:
            X - a data frame with test data
            save - save prediction in local directory or not
            f_format - format of data saving (if save = True): 'csv' or 'excel' (default)
        Return:
            the numeric classes.
        '''
        assert f_format in {'excel', 'csv'}
        preds = pd.DataFrame(self.best_estimator_.predict(X))
        if save == True and f_format == 'csv':
            preds.to_csv('preds.csv')
        elif save == True and f_format == 'excel':
            preds.to_excel('preds.xlsx', sheet_name = 'preds')
        else:
            pass
        return self.best_estimator_.predict(X)

    
    def prediction_report(self, X, y, metric=mean_absolute_error, save=False):
        '''
        Prediction report with metric score and two plots.
        Args:
            X - test data
            y - true values
            metric function (from sklearn package, default = mean_absolute_error)
            save - whether to save the output plots in local directory or not
        Return:
            plots
            Prediction score
        '''
        import matplotlib.lines as mlines
        preds = self.best_estimator_.predict(X)
        
        fig, ax = plt.subplots(figsize = (7, 5))
        plt.title('Predicted values')
        plt.scatter(y, preds)
        line = mlines.Line2D([0, 1], [0, 1], color = 'red')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        if save == True:
            plt.savefig('Preds_values.png', dpi = 200)
        plt.show()
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 7), dpi = 100)
        ax1.set_title('Residuals')
        ax1.scatter(y, y - preds)
        ax1.set_xlabel('True values')
        ax1.set_ylabel('Residuals')
        
        ax2.set_title('Residuals (QQplot)')
        sm.qqplot(preds, line = '45', fit = True, ax = ax2)
        if save == True:
            plt.savefig('Residuals.png', dpi = 200)
        plt.show()
        
        return print('Prediction score ({}): {}'.format(str(metric.__name__), 
                                                        metric(y, preds)))


class Clusterer:
    '''
    Unsupervised clustering algorithm.
    Args:
        clusterer - 'kmeans' (default) or 'DBSCAN'
        is_pca - if True, features will be transform with PCA (default = False)
        n_pca_comp - the number of components for PCA (if 'is_pca' = True) (default = 2)
        centroids_range - the range of centroids (K-means method) that the algorithm will work with (default = (2, 11))
        eps_range - the range of epsilons (DBSCAN method) that the algorithm will work with (default = (0.01, 1.01))
        random_state - a random_state parameter (default = 0)
    '''
    def __init__(self, clusterer='kmeans', is_pca=False, n_pca_comp=2,
                 centroids_range=(2, 11), eps_range=(0.01, 1.01),
                 random_state=0):
        self.clusterer = clusterer
        self.is_pca = is_pca
        self.pca_comp = n_pca_comp
        self.centroids_range = centroids_range
        self.eps_range = eps_range
        self.random_state = random_state
        
    def fit(self, X, save=False, f_format='excel'):
        '''
        Args:
            X - a data frame with features.
            save - save fit scores data and plots in local directory or not
            f_format - format of data saving (if save = True): 'csv' or 'excel' (default)
        Performs the data preprocessing and сalculates the quality of the model for a given range of parameters.
        Return:
            self.X - preprocessed input data (after scaling and PCA (if 'is_pca' = True))
            self.scores - quality values for all models
            a visualization of model quality changes
        '''
        assert f_format in {'excel', 'csv'}
        if X.isnull().sum().sum() > 0:
            X = X.dropna()
            print('Data has NaN values! Some rows were dropped!')
            
        # Preprocessing
        if self.is_pca == True:
            preprocessor = Pipeline([('scaler', StandardScaler()),
                                     ('PCA', PCA(n_components = self.pca_comp, 
                                                 random_state = self.random_state))])
        else:
            preprocessor = Pipeline([('scaler', StandardScaler())])
        self.X = preprocessor.fit_transform(X)
        
        if self.clusterer == 'kmeans':
            print('Using KMeans...')
            K = range(self.centroids_range[0], self.centroids_range[1])
            scores = pd.DataFrame({'k': K,
                                   'score': np.zeros(len(K))})
            for idx, k in enumerate(scores.k):
                kmeans = KMeans(init = 'k-means++', n_clusters = k,
                                n_init = 50, max_iter = 500,
                                random_state = self.random_state)
                kmeans.fit(self.X)
                scores.iloc[idx, 1] += kmeans.inertia_
            self.scores = scores
            print('Done!')
            
            plt.style.use("fivethirtyeight")
            plt.figure(figsize = (12, 5), dpi = 100)
            plt.plot(scores.k, scores.score)
            plt.xticks(scores.k)
            plt.xlabel("Number of Clusters")
            plt.ylabel("Sum of squared distances")
            
            if save == True:
                plt.savefig('KMeans_SSD.png', dpi = 200, bbox_inches = "tight")
            plt.show()
        else:
            print('Using DBSCAN...')
            eps = np.arange(self.eps_range[0], self.eps_range[1], 0.01)
            scores = pd.DataFrame({'eps': eps,
                                   'score': np.zeros(len(eps))})
            for idx, e in enumerate(scores.eps):
                dbscan = DBSCAN(eps = e)
                dbscan.fit(self.X)
                try:
                    scores.iloc[idx, 1] += silhouette_score(self.X, dbscan.labels_)
                except ValueError:
                    pass
            self.scores = scores
            print('Done!')
            
            plt.style.use("fivethirtyeight")
            plt.figure(figsize = (12, 5), dpi = 100)
            plt.plot(scores.eps, scores.score)
            plt.xticks(np.arange(self.eps_range[0], self.eps_range[1],
                                 (self.eps_range[1] - self.eps_range[0]) / 10))
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Coefficient")
            
            if save == True:
                plt.savefig('DBSCAN_SC.png', dpi = 200, bbox_inches = "tight")
            plt.show()
            
        if save == True and f_format == 'csv':
            pd.DataFrame(self.X).to_csv('data_preprocessed.csv')
            self.scores.to_csv('scores.csv')
        elif save == True and f_format == 'excel':
            pd.DataFrame(self.X).to_excel('data_preprocessed.xlsx', sheet_name = 'data_preprocessed')
            self.scores.to_excel('scores.xlsx', sheet_name = 'scores')
        else:
            pass
        
    def predict(self, k=None, eps=None, min_samples=5,
                save=False, f_format='excel'):
        '''
        Class prediction.
        Args:
            k - the number of K (for K-means)
            eps - epsilon parameter (for DBSCAN)
            min_samples - the number of samples in a neighborhood for a point (for DBSCAN) (default = 5)
            save - save prediction in local directory or not
            f_format - format of data saving (if save = True): 'csv' or 'excel' (default)
        Return:
            the numeric classes
            if 'is_pca' = True or the number of features = 2, also return plot with labeled data
        '''
        assert f_format in {'excel', 'csv'}
        if self.clusterer == 'kmeans':
            print('Using KMeans...')
            kmeans = KMeans(init = 'k-means++', n_clusters = k,
                            n_init = 50, max_iter = 500,
                            random_state = self.random_state)
            kmeans.fit(self.X)
            print('Done!')
            if self.is_pca == True or self.X.shape[1] == 2:
                plt.style.use("fivethirtyeight")
                plt.figure(figsize = (5, 5), dpi = 100)
                sns.scatterplot(self.X[:, 0], self.X[:, 1], 
                                hue = kmeans.labels_, palette = 'plasma')
                if save == True:
                    plt.savefig('KMeans_clusters.png', dpi = 200, 
                                bbox_inches = "tight")
                plt.show()
            
            if save == True and f_format == 'csv':
                pd.DataFrame(kmeans.labels_).to_csv('kmeans_labels.csv')
            elif save == True and f_format == 'excel':
                pd.DataFrame(kmeans.labels_).to_excel('kmeans_labels.xlsx', 
                                                      sheet_name = 'kmeans_labels')
            else:
                pass
            
            return kmeans.labels_
        else:
            print('Using DBSCAN...')
            dbscan = DBSCAN(eps = eps, min_samples = min_samples)
            dbscan.fit(self.X)
            print('Done!')
            if self.is_pca == True or self.X.shape[1] == 2:
                plt.style.use("fivethirtyeight")
                plt.figure(figsize = (5, 5), dpi = 100)
                sns.scatterplot(self.X[:, 0], self.X[:, 1], 
                                hue = dbscan.labels_, palette = 'plasma')
                if save == True:
                    plt.savefig('DBSCAN_clusters.png', dpi = 200,
                                bbox_inches = "tight")
                plt.show()
            
            if save == True and f_format == 'csv':
                pd.DataFrame(dbscan.labels_).to_csv('dbscan_labels.csv')
            elif save == True and f_format == 'excel':
                pd.DataFrame(dbscan.labels_).to_excel('dbscan_labels.xlsx', 
                                                      sheet_name = 'dbscan_labels')
            else:
                pass
            
            return dbscan.labels_


class KNN:
    '''
    Nearest Neighbors algorithm for the similarity finding.
    Args:
        n_neighbors - Number of neighbors to use (default - 2)
        missing_values - approach to missing values: "fill_zero"
    '''
    def __init__(self, n_neighbors=2, missing_values='fill_zero'):
        assert missing_values in {'fill_zero'}
        self.n_neighbors = n_neighbors
        self.missing_values = missing_values
        
    def fit(self, X, save=False):
        '''
        Args:
            X - a data frame
            save - save results in the local directory or not
        Return:
            self_distances_
            self_indices_
            model_
            fit_data_
        '''
        if X.isnull().sum().sum() > 0:
            if self.missing_values == 'fill_zero':            
                X = X.fillna(0)
                print('Data has missing values! Some values were filled by 0!')
            else:
                raise ValueError("Data has missing values!")
        
        model = NearestNeighbors(n_neighbors = self.n_neighbors).fit(X)
        self_distances, self_indices = model.kneighbors(X)
        self.self_distances_ = self_distances
        self.self_indices_ = self_indices
        self.model_ = model
        self.all_data_ = X
        
        if save == True:
            writer = pd.ExcelWriter('Self_NearestNeighbors.xlsx', engine = 'xlsxwriter')
            
            self.all_data_.to_excel(writer, sheet_name = 'data')
            pd.DataFrame(self.self_indices_).to_excel(writer, sheet_name = 'indices')
            pd.DataFrame(self.self_distances_).to_excel(writer, sheet_name = 'distances')
            pd.DataFrame(self.model_.kneighbors_graph(X).toarray()).to_excel(writer, sheet_name = 'graph')
            
            writer.save()
        else:
            pass
        
    def find_neighbors(self, X, save=False):
        '''
        Find nearest neighbors for the subset.
        Args:
            X - a data frame
            save - save results in the local directory or not
        Return:
            an array with neighbors' indices.
        '''
        if X.isnull().sum().sum() > 0:
            raise ValueError("Data has missing values!")

        self.distances, self.indices = self.model_.kneighbors(X)
        
        if save == True:
            writer = pd.ExcelWriter('NearestNeighbors.xlsx', engine = 'xlsxwriter')
            
            self.all_data_.to_excel(writer, sheet_name = 'all_data')
            X.to_excel(writer, sheet_name = 'subset_data')
            pd.DataFrame(self.indices).to_excel(writer, sheet_name = 'indices')
            pd.DataFrame(self.distances).to_excel(writer, sheet_name = 'distances')
            pd.DataFrame(self.model_.kneighbors_graph(X).toarray()).to_excel(writer, sheet_name = 'graph')
            
            writer.save()
        else:
            pass
        
        return self.indices
