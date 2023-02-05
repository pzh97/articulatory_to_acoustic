from data import *
import os
import time
import torch
from torch.autograd import Variable
from model import *

if __name__ == '__main__':
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
    #68.12165307998657 s
    m = np.c_[a, p]
    #m.shape=(1654242, 24)including labels
    indices = np.random.permutation(m.shape[0])
    train_size = int(0.8*(m.shape[0]))
    training_idx, test_idx = indices[:train_size], indices[train_size:]
    training, testing = m[training_idx, :], m[test_idx, :]
    #print(training.shape)=(1323393, 24)
    #print(test.shape)=(330849, 24)
    idx_in_columns = [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22]
    train = training[:, idx_in_columns]
    labels_training = training[:, (training.shape[1]-1)]
    #labels_training = labels_training[:, np.newaxis]
    #print(labels_training.shape)=(1323393, 1)
    train_mean = np.mean(train, axis=0)
    train_std = np.std(train, axis=0)
    train_zscore = (train-train_mean)/train_std
    #print(time.time()-tic)
    traindata = Data(train_zscore, labels_training)
    batch_size = 1000
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)
    network = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    epochs = 2
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
    PATH = './model.pth'
    torch.save(network.state_dict(), PATH)

    test = testing[:, idx_in_columns]
    labels_testing = testing[:, (testing.shape[1]-1)]
    test_mean = np.mean(test, axis=0)
    test_std = np.std(test, axis=0)
    test_zscore = (test-test_mean)/test_std
    testdata = Data(test_zscore, labels_testing)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=2)

    network = Network()
    network.load_state_dict(torch.load('./model.pth'))

    dataiter = iter(testloader)
    inputs, labels = next(dataiter)
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = network(inputs)
            __, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {len(testdata)} testdata: {100 * correct // total}%')
     #print(str(time.time() - tic) + ' s')
