#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py

# Copyright (c) 2019, Richard Gerum
#
# This file is part of the saenopy package.
#
# saenopy is free software: you can redistribute it and/or modify
# it under the terms of the MIT licence.
#
# saenopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the license
# along with saenopy. If not, see <https://opensource.org/licenses/MIT>

from setuptools import setup

setup(name='saenopy',
      version="0.7.4",
      description='Semi-elastic fiber optimisation in python.',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='MIT',
      packages=['saenopy', 'saenopy.gui'],
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      install_requires=[
          'numpy',
          'scipy',
          'tqdm',
          'qimage2ndarray',
          'pyvista',
          'pyvistaqt>=0.6.0',
          'imagecodecs',
          #'jointforces @ https://github.com/christophmark/jointforces/archive/master.zip'
      ],
      extras_require={
        'compilation':  ["numba"],
      },
      entry_points={
           'console_scripts': ['saenopy=saenopy.gui_master:main'],
      },
      include_package_data=True,
)
