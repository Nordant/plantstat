from setuptools import setup

setup(name = 'PlantStat',
      version = '0.3.0',
      description = 'Package for main statistics and AutoML in plant sciences',
      long_description = 'A package with a set of functions for fast and convenient statistical processing of experimental data. Also includes simple AutoML algorithms and CV of microscopic images. Designed for needs of the LPBPS (Laboratory of Physiology and Biochemistry of Plant Stress, Kharkiv, Ukraine).',
      url = 'https://github.com/Nordant/plantstat.git',
      author = 'Maksym Shkliarevskyi',
      author_email = 'maksym.shkliarevskyi@gmail.com',
      license = 'Apache License 2.0',
      packages = ['plantstat'],
      keywords = ['python', 'statistics', 'automl'],
      zip_safe = False,
      include_package_data = True)