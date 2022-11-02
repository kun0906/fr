""" Common functions and utilities

"""
# Author: kun.bj@outlook.com

import os
import pickle
import time
from datetime import datetime

import numpy as np


def check_path(dir_path):
    # if os.path.isfile(dir_path):
    # 	dir_path = os.path.dirname(dir_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def dump(data, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_file, 'wb') as out:
        pickle.dump(data, out)


def timer(func):
    # This function shows the execution time of the passed function
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        print(f'{func.__name__}() starts at {datetime.now()}')
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'{func.__name__}() ends at {datetime.now()}')
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def fmt(data, precision=3):
    def _format(data2):
        if type(data2) == np.array or type(data2) == list:
            res = np.asarray([_format(v) for v in data2])
        else:
            res = f'{data2:.{precision}f}'

        return res

    return _format(data)
