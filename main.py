from data import *
import os

ema_path = './mocha_timit/msak0_v1.1'
ema_files = []
lar_files = []
for data_file in sorted(os.listdir(ema_path)):
    if data_file.endswith('.ema'):
        ema_files.append(data_file)
    if data_file.endswith('.lar'):
        lar_files.append(data_file)
transcription= []
for grid in sorted(os.listdir('./corpus/textgrids')):
    if grid.endswith('.TextGrid'):
        transcription.append(grid)
error = 0
zipped = zip(ema_files, transcription, lar_files)
zipped_list = list(zipped)
#print(zipped_list)
error = 0
for f, s, l in zipped_list:
    error+=1
    data_reader('.', f, s, l)
print(error)

