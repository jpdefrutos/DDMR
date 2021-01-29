import os
import errno
import nibabel as nb
import numpy as np
import re

def try_mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError as err:
        if err.errno == errno.EEXIST:
            print("Directory " + dir + " already exists")
        else:
            raise ValueError("Can't create dir " + dir)
    else:
        print("Created directory " + dir)


def function_decorator(new_name):
    """"
    Change the __name__ property of a function using new_name.
    :param new_name:
    :return:
    """
    def decorator(func):
        func.__name__ = new_name
        return func
    return decorator
