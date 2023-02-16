# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:11:44 2023

@author: belie
"""

import random
import string

def get_random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def string2h5(hf, x, key):
    asciiList = [n.encode("ascii", "ignore") for n in x]
    hf.create_dataset(key, data=asciiList)