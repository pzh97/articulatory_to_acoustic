import struct
import numpy as np
from scipy.ndimage import gaussian_filter
import librosa
import re
import sys
import textgrids
import os
import spychhiker
import parselmouth
from itertools import groupby
np.set_printoptions(threshold=sys.maxsize)

script = []
phone = []
phone_value = {}
for transcription in sorted(os.listdir('./female_us/textgrids')):
    if transcription.endswith('.TextGrid'):
        script.append(transcription)
for t in script:
    grid = textgrids.TextGrid('./female_us/textgrids/' + t)
    for phon in grid['phones']:
        l = phon.text.transcode()
        phone.append(l)
phone_set = set(phone)
phone_list = list(phone_set)

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
    with open(filename + '/fsew0_v1.1/' + ema, 'rb') as f:
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

    phon_max = []
    count = 0
    labels = []

    grid = textgrids.TextGrid(filename + '/female_us/textgrids/' + segmentation)

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

    y, sr = librosa.load(filename + '/fsew0_v1.1/' + laryngograph, sr=1500)
    tlar = np.arange(len(y))/sr

    f0, tf0 = praat_get_pitch(signal2sound(y, sr))
    f0[f0 > 0] = 1

    seqs = [(key, len(list(val))) for key, val in groupby(f0)]
    seqs = [(key, sum(s[1] for s in seqs[:i]), len) for i, (key, len) in enumerate(seqs)]
    time_index = [[s[1], s[1]+s[2]-1] for s in seqs if s[0]==1]
    time_point = []
    for i in range(len(time_index)):
        for time in time_index[i]:
            time_point.append(tf0[time])
    time_point = [x for x in time_point if x<=time_vec[-1]]
    voicing_array = np.zeros(len(time_vec))
    voice_index = []
    for i in range(len(time_point)):
        voice_result = next(k for k, value in enumerate(time_vec) if time_point[i]<value)
        voice_index.append(voice_result)
    for i in range(0, (len(voice_index)-2), 2):
        voicing_array[voice_index[i]:voice_index[i+1]]=1
    conca = np.c_[mtx, voicing_array]
    return conca, output_array, time_vec




