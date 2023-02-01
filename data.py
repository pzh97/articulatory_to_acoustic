import struct
import numpy as np
from scipy.ndimage import gaussian_filter
import librosa
import re
import sys
import textgrids
import os
import spychhiker
import matplotlib.pyplot as plt
import parselmouth
from itertools import groupby
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
#ipa = 'a aj aw aː b bʲ c cʰ d dʒ dʲ e ej f fʲ h i iː j k kʰ l m mʲ m̩ n n̩ o ow p pʰ pʲ s t tʃ tʰ tʲ u uː v vʲ w z æ ç ð ŋ ɐ ɑ ɑː ɒ ɒː ɔ ɔj ə əw ɚ ɛ ɛː ɜ ɜː ɝ ɟ ɡ ɪ ɫ ɫ̩ ɱ ɲ ɹ ɾ ʃ ʉ ʉː ʊ ʎ ʒ ʔ θ d̪ t̪'
#IPA_list = list(ipa.split(" "))
#print(list(phone_set - set(IPA_list)))
print(len(phone_list))
print(phone_list)

for i in range(len(phone_set)):
    phone_value[phone_list[i]] = i

def signal2sound(y, sr):    
    sound = parselmouth.Sound(y)
    sound.sampling_frequency = sr
    return sound

def praat_get_pitch(sound, f0min=70, f0max=500):
    f0Obj = sound.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)
    f0 = f0Obj.selected_array['frequency']
    tf0 = f0Obj.xs()    
    return f0, tf0


def data_reader(filename, ema, segmentation, laryngograph):
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
    #data_header = file_read[:(length_header+1)]
    #print(data_header)
    """
    b'EST_File Track\nDataType binary\nNumFrames 1548\nByteOrder 01\nNumChannels 20\nEqualSpace 1\nBreaksPresent true\nCommentChar ;\n\nChannel_0 ui_x\nChannel_1 li_x\nChannel_2 ul_x\nChannel_3 ll_x\nChannel_4 tt_x\nChannel_5 ui_y\nChannel_6 li_y\nChannel_7 ul_y\nChannel_8 ll_y\nChannel_9 tt_y\nChannel_10 tb_x\nChannel_11 td_x\nChannel_12 v_x \nChannel_13 ****\nChannel_14 bn_x\nChannel_15 tb_y\nChannel_16 td_y\nChannel_17 v_y \nChannel_18 ****\nChannel_19 bn_y\nEST_Header_End\n
    sensor_we_need = ["li", "ul", "ll", "tt", "tb", "td", "v"]
    sensor_need_idx = [3, 4, 5, 6, 12, 13, 14]
    """
 
    nb_bytes = len(data_bytes)
    nb_v = int(nb_bytes/4)

    data = [struct.unpack('f', data_bytes[n*4:(n+1)*4])[0] for n in range(nb_v)]

    nb_col = 22
    nb_row = int(len(data)/nb_col)

    mtx = np.zeros((nb_row, nb_col))
    for n in range(nb_row):
        mtx[n, :] = data[n*nb_col:(n+1)*nb_col]
    
    time_vec = mtx[:, 0]
    #print(mtx.shape)
    #print(time_vec)
    #print(len(time_vec))

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

    output_array = output_array.reshape(-1, 1)

    y, sr = librosa.load(filename + '/mocha_timit/msak0_v1.1/' + laryngograph, sr=1500)
    tlar = np.arange(len(y))/sr
    #print(len(y))
    #print(sr)

    f0, tf0 = praat_get_pitch(signal2sound(y, sr))
    f0[f0 > 0] = 1

    seqs = [(key, len(list(val))) for key, val in groupby(f0)]
    seqs = [(key, sum(s[1] for s in seqs[:i]), len) for i, (key, len) in enumerate(seqs)]
    time_index = [[s[1], s[1]+s[2]-1] for s in seqs if s[0]==1]
    #print(time_index)
    time_point = []
    for i in range(len(time_index)):
        for time in time_index[i]:
            time_point.append(tf0[time])
    time_point = [x for x in time_point if x<=time_vec[-1]]
    #print(time_point)
    voicing_array = np.zeros(len(time_vec))
    voice_index = []
    for i in range(len(time_point)):
        voice_result = next(k for k, value in enumerate(time_vec) if time_point[i]<value)
        voice_index.append(voice_result)
    #print(voice_index)
    for i in range(0, (len(voice_index)-2), 2):
        voicing_array[voice_index[i]:voice_index[i+1]]=1
    conca = np.c_[mtx, voicing_array]
    #print(conca.shape)
    return conca, output_array, time_vec

#data_reader('.', 'fsew0_001.ema', 'fsew0_001.TextGrid', 'fsew0_001.lar')



