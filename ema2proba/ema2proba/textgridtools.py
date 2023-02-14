# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:49:23 2023

@author: belie
"""

import textgrids

def extract_grid(textgrid_file):
    return textgrids.TextGrid(textgrid_file)

def get_index(phon_max, i, time_vec):
    return next(k for k, value in enumerate(time_vec) if phon_max[i]<value) 
    
def grid_points(phon):
    return phon.text.transcode(), phon.xmin, phon.xmax

def grid2segphone(grid):
    pts = [grid_points(phon) for phon in grid["phones"]]
    return [[p[i] for p in pts] for i in range(3)]

def segphone2labels(labels, phon_min, phon_max, time_vec,
                    phone_value, bounds=None):
    
    output_array = [0 for t in time_vec]
    duration = [xmax - xmin for xmax, xmin in zip(phon_max, phon_min)]
    count = len(labels)
    index = [get_index(phon_max, i, time_vec) for i in range(count-2)]

    output_array[:index[0]] = phone_value[labels[0]]
    
    for n in range(len(index)-1):
        output_array[index[n]:index[n+1]] = phone_value[labels[n]]

    for n in range(len(time_vec)-index[-1]+1):
        output_array[index[-1]:] = phone_value[labels[-1]]  
        
    return output_array