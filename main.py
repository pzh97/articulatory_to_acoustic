from input_data import *
from output_data import *
import os

ema_path = './mocha_timit/msak0_v1.1'
files = []
data = []
for data_file in sorted(os.listdir(folder_path)):
    if data_file.endswith('.ema'):
        files.append(data_file)
for f in files:
    data.append(ema_reader(ema_path + '/' + f))
    
