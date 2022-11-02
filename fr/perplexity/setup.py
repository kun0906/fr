"""
	Compile inc_tsne_utils
    cd fr/perplexity
    pip install .   # method 1 (recommended)

    # method2
    # python setup.py build_ext --inplace     #(only compile locally, don't need to install by pip)
    # copy fr/perlexity/perplexity/_utils.cpython-310-darwin.so ./_utils.cpython-310-darwin.so
    # https://cython.readthedocs.io/en/latest/src/quickstart/build.html
    # from perplexity._utils import _binary_search_perplexity
"""
import os
import shutil

from Cython.Build import cythonize
from distutils.core import setup

import numpy

with open("README.md", 'r') as f:
    long_description = f.read()

APP_NAME = 'perplexity'
setup(
    version="0.0.1",
    name=APP_NAME,  # package name
    description='Find sigma by Binary search for a given perplexity',
    long_description=long_description,
    ext_modules=cythonize("perplexity/_utils.pyx"),
    include_dirs=[numpy.get_include()],
    #  zip_safe=False,
    # packages=find_packages(),
    # package_dir = {'perplexity': './'},  # the root directory of your package
    # packages=['perplexity'],
)

# clean data
build_dir = 'build'
if os.path.exists(build_dir): shutil.rmtree(build_dir)
# if os.path.exists(tests_dir): shutil.rmtree(tests_dir)
# if os.path.exists(build_dir): shutil.rmtree(build_dir)
if os.path.exists(f'{APP_NAME}.egg-info'): shutil.rmtree(f'{APP_NAME}.egg-info')