from setuptools import setup, find_packages
import plantstat

setup(name = 'PlantStat',
      version = plantstat.__version__,
      description = 'Package for main statistics and AutoML in plant sciences',
      long_description = 'A package with a set of functions for fast and convenient statistical processing of experimental data. Also includes simple AutoML algorithms and CV of microscopic images. Designed for needs of the LPBPS (Laboratory of Physiology and Biochemistry of Plant Stress, Kharkiv, Ukraine).',
      url = 'https://github.com/Nordant/plantstat',
      author = 'Maksym Shkliarevskyi',
      author_email = 'maksym.shkliarevskyi@gmail.com',
      license = 'Apache License 2.0',
      keywords = ['python', 'statistics', 'automl'],
      zip_safe = False,
      install_requires=['pandas', 'numpy', 'matplotlib', 'scipy', 'scikit-learn', 'seaborn',
                        'statsmodels', 'torch', 'torchvision', 'googledrivedownloader', 'opencv-python'],
      python_requires='>=3',      
      packages = find_packages())