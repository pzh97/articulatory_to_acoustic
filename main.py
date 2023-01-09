from data import *
import os
import time
import torch
from torch.autograd import Variable

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
print("start!")
epochs = 400
dtype = torch.FloatTensor
articulatory_d = Variable(torch.from_numpy(a).type(dtype), requires_grad = False)
phonetic_d = Variable(torch.from_numpy(p).type(dtype), requires_grad = False)

model = torch.nn.Sequential(
    torch.nn.Linear(23, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1654242),
    torch.nn.ReLU(),
    )
loss_fn = torch.nn.MSELoss()
for ix in range(epochs):
    p_hat = model(articulatory_d)
    loss_var = 0.5 * loss_fn(p_hat, phonetic_d)
    loss[ix] = loss_var.item()

    model.zero_grad()
    loss_var.backward()
    for param in model.parameters():
        param.data = param.data - param.grad.data
print(str(time.time() - tic) + ' s')

"""
# Ploting loss vs epochs
plt.figure()
ix = np.arange(epochs)
plt.plot(ix, loss)

# Training Accuracy
p_hat = model(X)
p_tmp = torch.max(p_hat, dim=1)[1]
phat = phone_value[p_tmp.data.numpy()]
acc = np.mean(1 * (phat == phonetic_d))
print('Training Accuracy: ' + str(acc*100))
"""
