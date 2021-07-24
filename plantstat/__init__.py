__version__ = '0.4.0'
__description__ = 'Package for main statistics and AutoML in plant sciences'
__author__ = 'Maksym Shkliarevskyi (maksym.shkliarevskyi@gmail.com)'
__all__ = ['vision']

from .stat_func import Variable_Analyzer
from .ML_func import AutoML_Classifier, AutoML_Regressor, Clusterer, KNN

