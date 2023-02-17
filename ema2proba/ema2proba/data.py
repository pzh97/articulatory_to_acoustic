import struct
import numpy as np
from scipy.ndimage import gaussian_filter
import librosa
import sys
import textgrids
from os import path, listdir, mkdir
import parselmouth
from itertools import groupby
np.set_printoptions(threshold=sys.maxsize)
import h5py
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from .filetools import read_data_file, read_file_list
from .utils import string2h5


def add_delta_features(X, delta, y, verbosity=True):
    nb_smpl, nb_feat = X.shape
    new_x = X*1    
    for n in tqdm(range(1, delta+1), desc="Delta features", 
                  disable=verbosity):
        xm1 = np.vstack((np.zeros((n, nb_feat)), new_x[:-n, :]))
        xp1 = np.vstack((new_x[n:, :], np.zeros((n, nb_feat))))
        X = np.hstack((X, xm1, xp1))
    return X[delta:-delta, :], y[delta:-delta]

def build_feature_matrix(feature_dir, delta=0, discard_bounds=None, 
                         verbosity=0):
    features = []
    output_class = []
    list_files = listdir(feature_dir)
    if verbosity >= 1:
        disable_list = False
    else:
        disable_list = True
    for l in tqdm(list_files, desc="Feature files", disable=disable_list):
        curr_file = path.join(feature_dir, l)
        with h5py.File(curr_file, "r") as hf:
            X = hf["features"][()]
            y = [s.decode("ascii") for s in hf["class_vector"][()]]
            if discard_bounds is not None:
                labels = [s.decode("ascii") for s in hf["labels"][()]]
                sbs = hf["segment_bounds"][()]
                tvec = hf["time_vector"][()]
        if delta > 0:
            if verbosity == 2:
                disable_delta = False
            else:
                disable_delta = True
            X, y = add_delta_features(X, delta, y, verbosity=disable_delta)
        if discard_bounds is not None:
            if delta > 0:
                tvec = tvec[delta:-delta]
            idx_keep = []
            for n in range(len(labels)):
                
                idx = [i for i in range(len(tvec)) if tvec[i] >= sbs[n, 0] and tvec[i] <= sbs[n, 1]]
                N = len(idx)
                N1 = int(N*(1-discard_bounds)/100/2)
                N2 = int(N*discard_bounds/100)
                if N > 3:
                    idx_keep += list(range(idx[0] + N1, 
                                           idx[0] + N1 + N2 + 1))
            idx_keep = np.unique([i for i in idx_keep if i < X.shape[0]]).tolist()
            
            X = X[idx_keep, :]
            y = [y[i] for i in idx_keep]
            
        output_class += y  
        features += X.tolist()        
   
    return np.array(features), output_class

def class_balance(labels):
    classes = np.unique(labels)
    return {c: labels.count(c) for c in classes}

def data_noise_augmentation(nb_aug, X_train, y_train, sigma_noise, 
                            isVoicing=True):
    copy_x = X_train * 1  
    copy_y = y_train * 1
    for n in tqdm(range(nb_aug), desc="Data augmentation"):
        nb_sampl, nb_feat_input = X_train.shape
        if isVoicing:
            Xaugment = np.hstack((copy_x[:,:-1] + sigma_noise * np.random.randn(nb_sampl, 
                                                                            nb_feat_input-1),
                                  copy_x[:, -1].reshape(-1, 1)))
        else:
            Xaugment = copy_x + sigma_noise * np.random.randn(nb_sampl, 
                                                                 nb_feat_input)
            
        X_train = np.vstack((X_train, Xaugment))
        y_train = np.vstack((y_train.reshape(-1, 1), copy_y.reshape(-1, 1))).reshape(-1)
    return X_train, y_train

def data_reader(ema_file, segmentation_file, egg_file, data_dir, parameters):

    time_vec, mtx =  extract_ema_features(ema_file, idx_keep=[3, 4, 5, 6, 8, 9, 10, 11,
                                12, 13, 14, 17, 18, 19, 22])

    phone_value = extract_phones(data_dir)
    output_array, idx_to_keep = read_txtgrid(segmentation_file, 
                                        time_vec, phone_value, parameters)

    voicing_array = read_egg_file(egg_file, time_vec)

    conca = np.c_[mtx, voicing_array]
    return conca, output_array, time_vec

def decode_label(labels):
    return [l.decode("ascii") for l in labels]

def extract_ema_features(ema_file, idx_keep=[3, 4, 5, 6, 8, 9, 10, 11,
                            12, 13, 14, 17, 18, 19]):
    
    with open(ema_file, 'rb') as f:
        
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

    return mtx[:, 0], mtx[:, idx_keep]

def extract_egg_features(egg_file, time_vec):
    
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

def extract_labels(time_vec, seg_file):
    
    insts, labels = read_segmentation_file(seg_file)

    phon_max = insts[:, -1]
    count = len(labels)
    
    output_array = [[]]*len(time_vec)
    index = []
    for i in range(count-2):
        result = next(k for k, value in enumerate(time_vec) if phon_max[i]<value)
        index.append(result)

    for n in range(index[0]):
        output_array[n] = labels[0]

    for n in range(len(index)-1):
        for idx in range(index[n], index[n+1]+1):
            output_array[idx] = labels[n]

    for n in range(len(time_vec)-index[-1]):
        output_array[index[-1]+n] = labels[-1]

    return output_array, insts, labels

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
    return {phone_list[i]: i for i in range(len(phone_set))}

def merge_phoneme(phoneme_merging, phonemes, labels, verbosity=1):
    keep = [p.split("-")[0] for p in phoneme_merging]
    merge = [p.split("-")[1] for p in phoneme_merging]
    if verbosity > 0:
        print("Merging phonemes...", flush=True)
    new_phonemes = [p for p in phonemes]
    
    for u, v in zip(keep, merge):
        if verbosity > 0:
            print("Merging " + v + " with " + u + " (keep " + u + ")", 
                  flush=True)
        if v in phonemes and u in phonemes:
            idx_v = phonemes.index(v)
            idx_u = phonemes.index(u)
            labels[labels==idx_v] = idx_u
            new_phonemes.remove(v)
        elif verbosity > 0:
            print("One of the phone to merge does not exist => skipping merging", 
                  flush=True)
    new_labels = np.array([l for l in labels])
    for n, p in enumerate(new_phonemes):
        idx_old = phonemes.index(p)
        idx = np.argwhere(labels==idx_old)[:, 0].astype(int)
        new_labels[idx] = n
    labels = new_labels
    phonemes = new_phonemes
    return labels, phonemes

def merge_phoneme_strings(phoneme_merging, labels, verbosity=1):
    keep = [p.split("-")[0] for p in phoneme_merging]
    merge = [p.split("-")[1] for p in phoneme_merging]
    if verbosity > 0:
        print("Merging phonemes...", flush=True)
    
    for u, v in zip(keep, merge):
        if verbosity > 0:
            print("Merging " + v + " with " + u + " (keep " + u + ")", 
                  flush=True)
        labels = [replace_phone(l, v, u) for l in labels]
        
    return labels

def read_segmentation_file(seg_file):
    lines = read_data_file(seg_file)        
    start, end, phone = [[l.split(" ")[n] for l in lines] for n in range(3)]
    return np.array([start, end]).astype(float).T, phone

def replace_phone(label, old, new):
    if label == old:
        label = new
    return label

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

def write_features_from_list(list_file, output_dir, verbosity=0, 
                             idx_keep=[3, 4, 5, 6, 8, 9, 10, 11,
                                       12, 13, 14, 17, 18, 19]):
    
    if verbosity == 0:
        v = True
    else:
        v = False
    utt_id, spk_id, emas, audios, eggs, segs = read_file_list(list_file)
    speakers = np.unique(spk_id).tolist()
    if not path.isdir(output_dir):
        mkdir(output_dir)
    for spk in speakers:
        speaker_dir = path.join(output_dir, spk)
        if not path.isdir(speaker_dir):
            mkdir(speaker_dir)
        idx = [i for i in range(len(emas)) if spk_id[i] == spk]
        for i in tqdm(idx, desc="Features from speaker " + spk, disable=v):
            feature_file = path.join(speaker_dir, utt_id[i] + ".h5")
            write_feature_file(feature_file, emas[i], eggs[i], segs[i], 
                               idx_keep=idx_keep)
        
    

def write_feature_file(feature_file, ema_file, egg_file, seg_file, idx_keep=[3, 4, 5, 6,
                                                              8, 9, 10, 11,
                                                              12, 13, 14, 17,
                                                              18, 19]):
    
    time_vec, mtx =  extract_ema_features(ema_file, idx_keep)

    voicing = extract_egg_features(egg_file, time_vec)
    mtx = np.c_[mtx, voicing]
    
    class_vector, insts, labels = extract_labels(time_vec, seg_file)
    with h5py.File(feature_file, "w") as hf:
        hf.create_dataset("features", data=mtx)
        hf.create_dataset("time_vector", data=time_vec)
        hf.create_dataset("segment_bounds", data=insts)
        string2h5(hf, class_vector, "class_vector")
        string2h5(hf, labels, "labels")


