import struct
import numpy as np
from scipy.ndimage import gaussian_filter
import librosa
import re
import sys
import textgrids

# length_header = 446
file_name = './mocha_timit/msak0_v1.1/msak0_001.ema'
with open(file_name, 'rb') as f:
    file_read = f.read()#[:446]
length_header = 0
isContinue = True
while isContinue:    
    header = "".join([struct.unpack('c', file_read[n:n+1])[0].decode("ascii") for n in range(length_header)])
    if header.endswith("End\n"):     
        isContinue = False
    else:
        length_header += 1

sensor_we_need = ["li", "ul", "ll", "tt", "tb", "td", "v"]

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
ema_1 = mtx[:, 11]*1e-5
ema_smooth = gaussian_filter(ema_1, sigma=2)

import matplotlib.pyplot as plt
plt.close("all")

plt.figure()
plt.plot(time_vec, ema_1)
plt.plot(time_vec, ema_smooth, '--r')

y, sr = librosa.load("./mocha_timit/msak0_v1.1/msak0_001.lar", sr=500)

order = r'[0-9]'
with open('./mocha_timit/msak0_v1.1/mocha-timit.txt', 'r') as t:
    lines = []
    for line in t:
        new_line = re.sub(order, '', line).replace('. ', '').replace('.', '')
        lines.append(new_line)

with open('transcript.txt', 'w') as w:
    w.write(lines[0])

phon_max = []
count = 0
labels = []
for arg in sys.argv[1:]:
    try:
        grid = textgrids.TextGrid(arg)
    except:
        continue

    for phon in grid['phones']:
        count+=1
        label = phon.text.transcode()
        labels.append(label)
        print('"{}";{};{}'.format(label, phon.xmin, phon.xmax))
        phon_max.append(phon.xmax)

print(phon_max)
print(count)
output_array = np.zeros(len(time_vec))
phone_value = {}
print(labels)

for i in range(count):
    phone_value[labels[i]] = i
print(phone_value)
index = []
for i in range(count-2):
    result = next(k for k, value in enumerate(time_vec) if phon_max[i]<value)
    index.append(result)

print(index)

for n in range(index[0]):
    output_array[0:index[0]] = phone_value[labels[0]]
print(output_array[354])

n = 1
for n in range(index[n+1]-index[n]+1):
    if n < 12:
        output_array[index[n]:index[n+1]] = phone_value[labels[n]]
print(output_array[421])

for n in range(len(time_vec)-index[-1]+1):
    output_array[index[-1]:] = phone_value[labels[-1]]
print(output_array[-1])
print(len(output_array))
print(len(time_vec))
