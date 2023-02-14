import struct
import numpy as np
from scipy.ndimage import gaussian_filter
import librosa
import sys
import textgrids
from os import path, listdir
import parselmouth
from itertools import groupby
np.set_printoptions(threshold=sys.maxsize)
import h5py
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def class_balance(labels):
    classes = np.unique(labels)
    return {c: labels.count(c) for c in classes}

def data_reader(ema_file, segmentation_file, egg_file, data_dir, parameters):

    mtx = read_ema_file(ema_file)
    time_vec = mtx[:, 0]

    phone_value = extract_phones(data_dir)
    output_array, idx_to_keep = read_txtgrid(segmentation_file, 
                                        time_vec, phone_value, parameters)

    voicing_array = read_egg_file(egg_file, time_vec)

    conca = np.c_[mtx, voicing_array]
    return conca, output_array, time_vec

def decode_label(labels):
    return [l.decode("ascii") for l in labels]

def extract_phones(data_dir):
    script = []
    phone = []
    for transcription in sorted(listdir(data_dir)):
        if transcription.endswith('.TextGrid'):
            script.append(transcription)
    for t in script:
        grid = textgrids.TextGrid(path.join(data_dir, t))
        for phon in grid['phones']:
            l = phon.text.transcode()
            phone.append(l)
    phone_set = set(phone)
    phone_list = list(phone_set)
    return {phone_list[i]: i for i in range(len(phone_set)) }

def sample_oversampling(data_mtx, num_class, classes, n_neighbors=5):
    idx = [x for x in range(len(classes)) if classes[x]==num_class]
    X = data_mtx[idx, :]
    idx_c = np.random.choice(idx)
    feat_c = data_mtx[idx_c, :]
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)
    neigh_idx = np.random.choice(neigh.kneighbors(feat_c.reshape(1, -1))[1].squeeze())
    feat_n = data_mtx[neigh_idx, :]
    return (feat_c - np.random.rand() * (feat_n - feat_c)).reshape(1, -1)

def praat_get_pitch(sound, f0min=70, f0max=500):
    f0Obj = sound.to_pitch(pitch_floor=f0min, pitch_ceiling=f0max)
    f0 = f0Obj.selected_array['frequency']
    tf0 = f0Obj.xs()
    return f0, tf0

def read_bunch(ema_dir, seg_dir, parameters=None, 
               idx_keep=[3, 4, 5, 6, 8, 9, 10, 11,
                                           12, 13, 14, 17, 18, 19, 22]):

    ema_files = []
    lar_files = []
    for data_file in sorted(listdir(ema_dir)):
        if data_file.endswith('.ema'):
            ema_files.append(path.join(ema_dir, data_file))
        if data_file.endswith('.lar'):
            lar_files.append(path.join(ema_dir, data_file))
    transcription= []
    for grid in sorted(listdir(seg_dir)):
        if grid.endswith('.TextGrid'):
            transcription.append(path.join(seg_dir, grid))
    zipped = zip(ema_files, transcription, lar_files)
    zipped_list = list(zipped)
    articulatory_data = []
    phonetic_data = []
    for f, s, l in tqdm(zipped_list, desc="reading files"):
        articulatory, phonetic, time_vec = data_reader(f, s, l, seg_dir, 
                                                       parameters)
        articulatory_data.append(articulatory)
        phonetic_data.append(phonetic)
    a = np.vstack(articulatory_data)
    p = np.vstack(phonetic_data)
    phone_value = extract_phones(seg_dir)
    return a[:, idx_keep], p, phone_value

def read_egg_file(egg_file, time_vec):

    y, sr = librosa.load(egg_file, sr=1500)

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
    return voicing_array

def read_ema_file(filename):
    with open(filename, 'rb') as f:
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

    return mtx

def read_file(file, keys):
    with h5py.File(file, "r") as hf:
        return [hf[key][()] for key in keys]

def read_txtgrid(segmentation_file, time_vec, phone_value, parameters=None):

    phon_max = []
    count = 0
    labels = []
    output_array = np.zeros(len(time_vec))
    grid = textgrids.TextGrid(segmentation_file)

    for phon in grid['phones']:
        count+=1
        label = phon.text.transcode()
        labels.append(label)
        #print('"{}";{};{}'.format(label, phon.xmin, phon.xmax))
        phon_max.append(phon.xmax)

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

    return output_array.reshape(-1, 1), labels

def signal2sound(y, sr):
    sound = parselmouth.Sound(y)
    sound.sampling_frequency = sr
    return sound

def undersampling(X, y, threshold=None):
    classes = np.unique(y)
    nb_class = np.sort([y.tolist().count(c) for c in classes])
    nb_min_class = nb_class[0]
    if threshold is None:
        threshold = nb_min_class
    else:
        i = 1
        while nb_min_class < threshold:
            nb_min_class = nb_class[i]
            i += 1
    for c in classes:
        nb_smpl = len(y)
        idx = [i for i in range(nb_smpl) if y[i] == c]
        if len(idx) > nb_min_class and len(idx) > threshold:
            nb_to_remove = len(idx) - nb_min_class
            np.random.shuffle(idx)
            idx_to_remove = idx[:nb_to_remove]
            X = np.delete(X, idx_to_remove, axis=0)
            y = np.delete(y, idx_to_remove, axis=0)
    return X, y


