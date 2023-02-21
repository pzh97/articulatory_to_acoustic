import struct
import numpy as np
import re
import sys
import os
from itertools import groupby
from tqdm import tqdm
#np.set_printoptions(threshold=sys.maxsize)


def get_phones_time(f, t, l):
    count = 0
    with open(f, 'r') as f:
        while True:
            #count+=1
            line = f.readline().split()
            if not line:
                break
            count+=1
            label = line[2]
            time = line[1]
            t.append(time)
            l.append(label)
    return t, l, count

def ema_reader(filename, ema):
    with open(filename + '/fsew0_v1.1/' + ema, 'rb') as f:
        file_read = f.read()
    length_header = 0
    isContinue = True
    while isContinue:
        header = "".join([struct.unpack('c', file_read[n:n+1])[0].decode("ascii") for n in range(length_header)])
        if header.endswith("End\n"):
            isContinue = False
        else:
            length_header +=1
        
    data_bytes = file_read[length_header:]
 
    nb_bytes = len(data_bytes)
    nb_v = int(nb_bytes/4)

    data = [struct.unpack('f', data_bytes[n*4:(n+1)*4])[0] for n in range(nb_v)]

    nb_col = 22
    nb_row = int(len(data)/nb_col)

    mtx = np.zeros((nb_row, nb_col))
    for n in range(nb_row):
        mtx[n, :] = data[n*nb_col:(n+1)*nb_col]
    
    time_vec = mtx[:, 0]
    return time_vec, mtx

def build_labels(time_vec, time, phone_value, count, l):
    output_array = np.zeros(len(time_vec))
    index = []
    for i in range(count-2):
        result = next(k for k, value in enumerate(time_vec) if time[i]<value)
        index.append(result)

    for n in range(index[0]):
        output_array[0:index[0]] = phone_value[l[0]]
     
    for n in range(len(index)-1):
        output_array[index[n]:index[n+1]] = phone_value[l[n]]

    for n in range(len(time_vec)-index[-1]+1):
        output_array[index[-1]:] = phone_value[l[-1]]

    output_array = output_array.reshape(-1, 1)
    return output_array

def add_delta_features(X, delta, y):
    nb_smpl, nb_feat = X.shape
    new_x = X*1    
    for n in tqdm(range(1, delta+1), desc="Delta features"):
        xm1 = np.vstack((np.zeros((n, nb_feat)), new_x[:-n, :]))
        xp1 = np.vstack((new_x[n:, :], np.zeros((n, nb_feat))))
        X = np.hstack((X, xm1, xp1))
    return X[delta:-delta, :], y[delta:-delta]



