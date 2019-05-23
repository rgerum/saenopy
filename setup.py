#!/usr/bin/env python
# -*- coding: utf-8 -*-
# setup.py

# Copyright (c) 2019, Richard Gerum
#
# This file is part of the saeno package.
#
# saeno is free software: you can redistribute it and/or modify
# it under the terms of the MIT licence.
#
# saeno is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the license
# along with saeno. If not, see <https://opensource.org/licenses/MIT>

from setuptools import setup

setup(name='saeno',
      version="0.9",
      description='Projects coordinates from 2D to 3D and can fit camera parameters',
      author='Richard Gerum',
      author_email='richard.gerum@fau.de',
      license='MIT',
      packages=['saeno'],
      install_requires=[
          'numpy',
          'scipy',
          'tqdm'
      ],
      extras_require={
        'compilation':  ["numba"],
      }
      )
