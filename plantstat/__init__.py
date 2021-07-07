__version__ = '0.3.0'
__description__ = 'Package for main statistics and AutoML in plant sciences'
__author__ = 'Maksym Shkliarevskyi (maksym.shkliarevskyi@gmail.com)'

from .stat_func import Variable_Analyzer
from .ML_func import AutoML_Classifier, AutoML_Regressor, Clusterer

# list of all modules available in the library
__all__ = ['vision']