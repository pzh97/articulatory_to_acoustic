from data import *
import os
import time
import torch
from torch.autograd import Variable
from model import *

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
#m.shape=(1654242, 24)including labels
t = m[:, 2:(m.shape[1])]
timit = torch.from_numpy(t)
t_input = timit[:, : (timit.shape[1]-1)]
#shape=(1654242, 21) excluding labels, time vector and participation information
t_output = timit[:, (timit.shape[1]-1)]
normalised_input = torch.nn.functional.normalize(t_input, p=2.0, dim=0, eps=1e-12)
normalised_input = torch.hstack([normalised_input, t_output.unsqueeze(1)])
idx_in_columns = [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19]
normalised_input = normalised_input[:, idx_in_columns]
print(normalised_input.shape)
dataset = torch.utils.data.TensorDataset(normalised_input, t_output)
train_size = int(0.8 * len(normalised_input))
test_size = len(normalised_input) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

"""
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=True)


optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  

accuracy = 0

for t in range(100):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    x, y = train_dataset[sample_idx]
    out = net(x.float())               
    loss = loss_func(out, y.long())   

    optimizer.zero_grad() 
    loss.backward()        
    optimizer.step()       

    if t % 2 == 0:
        prediction = torch.max(out, 0)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print(accuracy)
"""

