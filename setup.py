"""
Created on Thu Apr 5 2021
@author: Nordant
"""

from setuptools import setup
setup(name = 'PlantStat',
      version = '0.1.1',
      description = 'Package for main statistics in plant sciences',
      long_description = 'A package with a set of functions for fast and convenient statistical processing of experimental data. Designed for needs of the LPBPS (Laboratory of Physiology and Biochemistry of Plant Stress; Kharkiv, Ukraine).',
      url = 'https://github.com/Nordant/plantstat.git',
      author = 'Maksym Shkliarevskyi',
      author_email = 'maksym.shkliarevskyi@gmail.com',
      license = 'Apache License 2.0',
      packages = ['plantstat'],
      keywords = ['python', 'statistics'],
      zip_safe = False)