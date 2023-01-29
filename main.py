from data import *
import os
import time
import torch
from torch.autograd import Variable
import torchvision

tic = time.time()
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
#print(transcription)
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
#print(str(time.time() - tic) + ' s')
#68.12165307998657 s
m = np.c_[a, p]
#print(m.shape)
t = m[:, 2:(m.shape[1])]
timit = torch.from_numpy(t)
t_input = timit[:, : (timit.shape[1]-1)]
t_output = timit[:, (timit.shape[1]-1)]
mean, std = torch.mean(t_input, 0), torch.std(t_input, 0)
normalised_input = torchvision.transforms.Normalize(mean, std)

