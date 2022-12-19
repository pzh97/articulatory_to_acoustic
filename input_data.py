import struct
import numpy as np
from scipy.ndimage import gaussian_filter
import librosa
import re
import sys
import textgrids

def ema_reader(filename):
    global time_vec
    with open(filename, 'rb') as f:
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
    print(time_vec)

def transcription_generator(filename):
    order = r'[0-9]'
    with open(filename) as t:
        lines = []
        for line in t:
            new_line = re.sub(order, '', line).replace('. ', '').replace('.', '')
            lines.append(new_line)

    with open('transcript.txt', 'w') as w:
        w.write(lines[1])


ema_reader('./mocha_timit/msak0_v1.1/msak0_001.ema')
transcription_generator('./mocha_timit/msak0_v1.1/mocha-timit.txt')


