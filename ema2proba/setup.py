#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 07:53:09 2021

@author: benjamin
"""

from setuptools import setup, find_packages

setup(name='ema2proba',
      version='0.0.0',
        description='Python interface to train ema to probability models',
        url='https://git.ecdf.ed.ac.uk/belie/maedeep',
        author='Benjamin Elie',
        author_email='benjamin.elie@ed.ac.uk',
        license='Creative Commons Attribution 4.0 International License',
      packages=find_packages(),
      install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "tensorflow", 
        "keras",
        "scikit-learn",
	"librosa",
        "pytest",
	"praat-textgrids",
	"praat-parselmouth"
    ],
      zip_safe=False)
