# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:11:44 2023

@author: belie
"""

import random
import string
from importlib.resources import files


def get_random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def get_repo_path():
    return files("ema2proba").parent.parent

def string2h5(hf, x, key):
    asciiList = [n.encode("ascii", "ignore") for n in x]
    hf.create_dataset(key, data=asciiList)