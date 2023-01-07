from data import *
import os

path = './mocha_timit/msak0_v1.1'
ema_files = []
lar_files = []
for data_file in sorted(os.listdir(path)):
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
articulatory_data = []
phonetic_data = []
for f, s, l in zipped_list:
    articulatory, phonetic, time_vec = data_reader('.', f, s, l)
    articulatory_data.append(articulatory)
    phonetic_data.append(phonetic)
a = np.vstack(articulatory_data)
p = np.vstack(phonetic_data)
#print(a.shape)
#(1654242, 23)
#print(p.shape)
#(1654242, 1)
