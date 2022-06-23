#!/usr/bin/env python
# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2020 Lindsey Gray (FNAL), Benedikt Maier (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os.path
import six

from setuptools import find_packages
from setuptools import setup

def get_version():
    g = {}
    exec(open(os.path.join("deepjet_geometric", "version.py")).read(), g)
    return g["__version__"]


def get_description():
    description = open("README.rst", "rb").read().decode("utf8", "ignore")
    start = description.index(".. inclusion-marker-1-5-do-not-remove")
    stop = description.index(".. inclusion-marker-3-do-not-remove")

    #    before = ""
    #    after = """
    # Reference documentation
    # =======================
    # """

    return description[start:stop].strip()  # before + + after


INSTALL_REQUIRES = ['torch>=1.5.0',
                    'torch-geometric>=1.6.0' 
                    ]
EXTRAS_REQUIRE = {}

setup(name="deepjet-geometric",
      version=get_version(),
      packages=find_packages(exclude=["tests"]),
      scripts=[],
      include_package_data=True,
      description="A package for doing geometric deep learning on jets with pytorch-geometric",
      long_description=get_description(),
      author="Lindsey Gray (Fermilab)",
      author_email="lagray@fnal.gov",
      maintainer="Lindsey Gray (Fermilab)",
      maintainer_email="lagray@fnal.gov",
      url="https://github.com/lgray/deepjet-geometric",
      download_url="https://github.com/lgray/deepjet-geometric/releases",
      license="MIT",
      test_suite="tests",
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      setup_requires=["pytest-runner", "flake8"],
      tests_require=["pytest"],
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Intended Audience :: Machine Learning",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
      ],
      platforms="Any",
      )
