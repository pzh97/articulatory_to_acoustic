from data import *
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model import *

if __name__ == '__main__':
    path = './fsew0_v1.1'
    ema_files = []
    transcriptions = []
    phone = []
    phone_value = {}
    phone = []
    time_vec = []
    matrix = []
    time_point = []
    phone_labels = []
    count_list = []
    for data_file in sorted(os.listdir(path)):
        if data_file.endswith('.ema'):
            ema_files.append(data_file)
    for label_file in sorted(os.listdir(path)):
        if label_file.endswith('.lab'):
            transcriptions.append(label_file)
    for ema in ema_files:
        time_vector, mtx = ema_reader('.', ema)
        time_vec.append(time_vector) #reference time
        matrix.append(mtx)

    matrix = np.vstack(matrix)#ema matrix built
    for transcription in transcriptions:
        time_max = [] #phone time points
        labels = []
        count = 0
        t, l, count = get_phones_time('./fsew0_v1.1/'+ transcription, time_max, labels)
        phone.extend(l)
        time_point.append(t) #time points of phones in each file
        phone_labels.append(l) #labels for each file
        count_list.append(count) #count of labels for each file
    phone = set(phone)
    phone_list = list(phone)
    for i in range(len(phone)):
        phone_value[phone_list[i]] = i

    zipped_list = zip(time_vec, time_point, count_list, phone_labels)
    phonetic_data = []
    for tv, tp, c, ls in zipped_list:
        time_p =  (np.array(tp)).astype(float)
        output_labels =  build_labels(tv, time_p, phone_value, c, ls)
        phonetic_data.append(output_labels)
    phonetic_data = np.vstack(phonetic_data)
    m = np.c_[matrix, phonetic_data]
    np.save('data.npy', m)
"""
    #column 22nd is the voicing information
    idx_no_voicing = [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19]
    temp = m[:, idx_no_voicing]
    idx = (argrelextrema(temp, np.less, axis=0)[0]).flatten().tolist() + (argrelextrema(temp, np.greater, axis=0)[0]).flatten().tolist()
    idx = np.unique(idx)
    #keep 834190 rows
    print(len(idx))
    m = m[idx, :]
    indices = np.random.permutation(m.shape[0])
    train_size = int(0.8*(m.shape[0]))
    training_idx, test_idx = indices[:train_size], indices[train_size:]
    training, testing = m[training_idx, :], m[test_idx, :]
    idx_in_columns = [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22]
    train = training[:, idx_in_columns]
    print(train.shape)
    labels_training = training[:, (training.shape[1]-1)]
    #print(labels_training)
    train_mean = np.mean(train, axis=0)
    train_std = np.std(train, axis=0)
    train_zscore = (train-train_mean)/train_std
    traindata = Data(train_zscore, labels_training)
    batch_size = 1000
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2)
    network = Network()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    epochs = 10
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

    network.load_state_dict(torch.load('./model.pth'))

    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = network(inputs)
            __, predicted = torch.max(outputs.data, 1)
            #prob = F.softmax(outputs, dim=1)
            #top_p, top_class = prob.topk(1, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {len(testdata)} testdata: {100 * correct // total}%')
    """
