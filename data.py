import struct
import numpy as np
from scipy.ndimage import gaussian_filter
import librosa
import re
import sys
import textgrids
import os
np.set_printoptions(threshold=sys.maxsize)

script = []
phone = []
phone_value = {}
for transcription in sorted(os.listdir('./corpus/textgrids')):
    if transcription.endswith('.TextGrid'):
        script.append(transcription)
for t in script:
    grid = textgrids.TextGrid('./corpus/textgrids/' + t)
    for phon in grid['phones']:
        l = phon.text.transcode()
        phone.append(l)
phone_set = set(phone)
phone_list = list(phone_set)
ipa = 'a aj aw aː b bʲ c cʰ d dʒ dʲ e ej f fʲ h i iː j k kʰ l m mʲ m̩ n n̩ o ow p pʰ pʲ s t tʃ tʰ tʲ u uː v vʲ w z æ ç ð ŋ ɐ ɑ ɑː ɒ ɒː ɔ ɔj ə əw ɚ ɛ ɛː ɜ ɜː ɝ ɟ ɡ ɪ ɫ ɫ̩ ɱ ɲ ɹ ɾ ʃ ʉ ʉː ʊ ʎ ʒ ʔ θ'
IPA_list = list(ipa.split(" "))
#print(list(phone_set - set(IPA_list)))

for i in range(len(phone_set)):
    phone_value[phone_list[i]] = i

def data_reader(filename, ema, segmentation):
    with open(filename + '/mocha_timit/msak0_v1.1/' + ema, 'rb') as f:
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

    phon_max = []
    count = 0
    labels = []

    grid = textgrids.TextGrid(filename + '/corpus/textgrids/' + segmentation)

    for phon in grid['phones']:
        count+=1
        label = phon.text.transcode()
        labels.append(label)
        #print('"{}";{};{}'.format(label, phon.xmin, phon.xmax))
        phon_max.append(phon.xmax)

    output_array = np.zeros(len(time_vec))
    
    index = []
    for i in range(count-2):
        result = next(k for k, value in enumerate(time_vec) if phon_max[i]<value)
        index.append(result)


    for n in range(index[0]):
        output_array[0:index[0]] = phone_value[labels[0]]
     
    for n in range(len(index)-1):
        output_array[index[n]:index[n+1]] = phone_value[labels[n]]

    for n in range(len(time_vec)-index[-1]+1):
        output_array[index[-1]:] = phone_value[labels[-1]]
    return mtx, output_array

def transcription_generator(filename):
    order = r'[0-9]'
    with open(filename) as t:
        lines = []
        for line in t:
            new_line = re.sub(order, '', line).replace('. ', '').replace('.', '').replace('?', '').replace('\'', '').replace(',', '').replace('\"', '').replace('-', ' ')
            new_line = new_line.strip()
            if new_line:
                lines.append(new_line)
    path = r'./corpus'
    if not os.path.exists(path):
        os.makedirs(path)
                
    for i in range(len(lines)):
        with open("./corpus/msak0_%03d.txt"%(i+1), 'w') as w:
            w.write(lines[i])


#data_reader('.', 'msak0_268.ema', 'msak0_268.TextGrid')
#transcription_generator('./mocha-timit.txt')


