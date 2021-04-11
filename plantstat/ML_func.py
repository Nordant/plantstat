"""
Created on Sat Apr 10 2021
@author: Nordant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, classification_report, plot_confusion_matrix
from sklearn.metrics import mean_absolute_error


class AutoML_Classifier:
    '''
    AutoML algorithm for classification.
    Takes parameters of the estimated metric (default = balanced_accuracy) 
    the number of iterations of parameters search (default = 50),
    a random_state parameter (default = 0) and a number of cross_validation 
    repeats (default = 5).
    '''
    def __init__(self, scoring_func = 'balanced_accuracy', n_iter = 50,
                 random_state = 0, cv = 5):
        self.scoring_func = scoring_func
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        
    def fit(self, X, y):
        '''
        Takes predictors in pd.DataFrame format and predicted variable. 
        It selects an optimal machine learning algorithm and performs all 
        the data preprocessing necessary for this algorithm.
        Returns best_estimator_ and best_params_ required for prediction,
        and detailed cv results.
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
        
        # ALGORITHMS
        # Logistic regression
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner_strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [LogisticRegression()]
        })
        
        # K-nearest neighbors
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [KNeighborsClassifier()],
            'estimator__weights': ['uniform', 'distance'],
            'estimator__n_neighbors': np.arange(1, 20, 1)
        })

        # Random Forest
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [RandomForestClassifier(random_state = self.random_state)],
            'estimator__n_estimators': np.arange(5, 1000, 20),
            'estimator__criterion': ['gini', 'entropy']
        })

        # Gradient boosting
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [GradientBoostingClassifier(random_state = self.random_state)],
            'estimator__n_estimators': np.arange(5, 1000, 20),
            'estimator__learning_rate': np.linspace(0.01, 1.0, 30),
        })
        
        # XGBoost
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [XGBClassifier(random_state = self.random_state)],
            'estimator__n_estimators': np.arange(5, 1000, 20),
            'estimator__learning_rate': np.linspace(0.01, 1.0, 30),
        })

        # Decision tree
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [DecisionTreeClassifier(random_state = self.random_state)],
            'estimator__criterion': ['gini', 'entropy']
        })

        # Linear SVM
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean','median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [LinearSVC(random_state = self.random_state)],
            'estimator__C': np.arange(0.1, 1.1, 0.1),

        })
        
        # Search the best estimator
        search = RandomizedSearchCV(
                  model_pipeline,
                  optimization_grid,
                  n_iter = self.n_iter,
                  scoring = self.scoring_func, 
                  n_jobs = -1, 
                  random_state = self.random_state, 
                  verbose = 4,
                  cv = self.cv,
                  return_train_score = True)

        search.fit(X_train, y_train)
        
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_
        self.cv_results_ = search.cv_results_
        
        best_alg = str(self.best_pipeline['estimator']).split('(')[0]
        print('{} was used as the best algorithm!'.format(best_alg))
        
        
    def predict(self, X, y = None):
        '''
        Class prediction based on trained AutoML model.
        Returns the numeric classes.
        '''
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X, y = None):
        '''
        Class prediction based on trained AutoML model.
        Returns the probabilities.
        '''
        return self.best_estimator_.predict_proba(X)
    
    def classification_report(self, X, y, labels = None, cmap = 'inferno'):
        '''
        Prediction classification report.
        '''
        report = classification_report(y, self.best_estimator_.predict(X), 
                                       target_names = labels)
        
        plot_confusion_matrix(self.best_estimator_, X, y,
                              display_labels = labels, cmap = cmap)
        
        return print(report)
    
    
class AutoML_Regressor:
    '''
    AutoML algorithm for regression.
    Takes parameters of the estimated metric (default = neg_mean_squared_error) 
    the number of iterations of parameters search (default = 50),
    a random_state parameter (default = 0) and a number of cross_validation 
    repeats (default = 5).
    '''
    def __init__(self, scoring_func = 'neg_mean_squared_error', n_iter = 50,
                 random_state = 0, cv = 5):
        self.scoring_func = scoring_func
        self.n_iter = n_iter
        self.random_state = random_state
        self.cv = cv
        
    def fit(self, X, y):
        '''
        Takes predictors in pd.DataFrame format and predicted variable. 
        It selects an optimal machine learning algorithm and performs all 
        the data preprocessing necessary for this algorithm.
        Returns best_estimator_ and best_params_ required for prediction,
        and detailed cv results.
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
        
        # ALGORITHMS
        # Linear Regression
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner_strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [LinearRegression()]
        })
        
        # Lasso (L1)
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [Lasso(random_state = self.random_state)],
            'estimator__alpha': np.arange(0.001, 1.01, 0.05)
        })

        # Ridge (L2)
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [Ridge(random_state = self.random_state)],
            'estimator__alpha': np.arange(0.001, 1.01, 0.05)
        })
        
         # ElasticNet (L1+L2)
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [ElasticNet(random_state = self.random_state)],
            'estimator__alpha': np.arange(0.001, 1.01, 0.05)
        })

        # Random Forest Regressor
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [RandomForestRegressor(random_state = self.random_state)],
            'estimator__n_estimators': np.arange(5, 1000, 20),
            'estimator__criterion': ['gini', 'entropy']
        })

        # Gradient boosting
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [GradientBoostingRegressor(random_state = self.random_state)],
            'estimator__n_estimators': np.arange(5, 1000, 20),
            'estimator__learning_rate': np.linspace(0.01, 1.0, 30),
        })

        # XGBoost
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [XGBRegressor(random_state = self.random_state)],
            'estimator__n_estimators': np.arange(5, 1000, 20),
            'estimator__learning_rate': np.linspace(0.01, 1.0, 30),
        })

        # SVR
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean','median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [SVR()],
            'estimator__C': np.arange(0.1, 1.1, 0.1),
        })
        
        # Search the best estimator
        search = RandomizedSearchCV(
                  model_pipeline,
                  optimization_grid,
                  n_iter = self.n_iter,
                  scoring = self.scoring_func, 
                  n_jobs = -1, 
                  random_state = self.random_state, 
                  verbose = 4,
                  cv = self.cv,
                  return_train_score = True)

        search.fit(X_train, y_train)
        
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_
        self.cv_results_ = search.cv_results_
        
        best_alg = str(self.best_pipeline['estimator']).split('(')[0]
        print('{} was used as the best algorithm!'.format(best_alg))
        
        
    def predict(self, X, y = None):
        '''
        Prediction based on trained AutoML_Regression model.
        Returns the numeric values.
        '''
        return self.best_estimator_.predict(X)

    
    def prediction_report(self, X, y, metric = mean_absolute_error):
        '''
        Prediction report with metric score and two plots.
        Takes test data and metric function (from sklearn package).
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
        plt.show()
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 7), dpi = 100)
        ax1.set_title('Residuals')
        ax1.scatter(y, y - preds)
        ax1.set_xlabel('True values')
        ax1.set_ylabel('Residuals')
        
        ax2.set_title('Residuals (QQplot)')
        sm.qqplot(preds, line = '45', fit = True, ax = ax2)
        plt.show()
        
        return print('Prediction score ({}): {}'.format(str(metric.__name__), 
                                                        metric(y, preds)))
    