# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:34:27 2023

@author: belie
"""

from os import path, environ, listdir
import shutil
import ema2proba 
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import pickle

import time
plt.close("all")



def training_model(delta, speaker="female"):
    
    output_file = "model_delta_" + str(delta) + ".mlp"
    undersampling = 0
    phoneme_merging = ['sil-breath']
    phoneme_removing = None #['sil']
    phoneme_undersampling = {'sil': 50000}
    hdl = (100,)
    activation_function = 'tanh'
    num_epochs = 200
    batch_size = 1024
    model_architecture = "MLP"
    verbosity = 1
    prop_test = 10
    prop_valid = 5
    data_augmentation = 0
    sigma_noise = 0.01
    alpha = 0
    loss_function = None
    normalization = "zscore"
    if speaker == "female":
        feature_dir = path.join("..", "features", "fsew0_v1.1")
    else:
        feature_dir = path.join("..", "features", "msak0_v1.1")
        
    X, y = ema2proba.build_feature_matrix(feature_dir, delta=delta,
                                      discard_bounds=None, 
                                      verbosity=verbosity)
    
    if phoneme_merging is not None:
        y = ema2proba.merge_phoneme_strings(phoneme_merging, y, 
                                            verbosity=verbosity)        
    if phoneme_removing is not None: 
        X, y = ema2proba.remove_phoneme(X, y, phoneme_removing)
    if phoneme_undersampling is not None: 
        X, y = ema2proba.undersampling_phoneme(X, y, phoneme_undersampling)
    print("Speaker normalization...", flush=True)
    X, scaler = ema2proba.normalization(X, norm=normalization)    
    phonemes = [p for p in np.unique(y)] 
    print(phonemes)
    
    n = 0
    y = np.array(y).squeeze()
    for p in tqdm(phonemes, desc="assigning class number"):    
        for i in range(len(y)):
            if y[i] == p:
                y[i] = n
        n += 1
    y = y.astype(int)
    if undersampling > 0:
        print("Performing undersampling...")
        X, y = ema2proba.undersampling(X, y, threshold=undersampling)

    cl = np.unique(y)
    nb_cl = [np.count_nonzero(y == c) for c in cl]
    num_class = len(cl)

    print("Splitting train and test datasets...", flush=True)
    X_train, X_test, y_train, y_test = ema2proba.split_data(X, y, 
                                                  test_size=(prop_test+prop_valid)/100, 
                                                  random_state=42)

    nb_aug = data_augmentation
    nb_sampl, nb_feat_input = X_train.shape
    if nb_aug > 0:
        if verbosity > 0:
            print("Performing data augmentation on training dataset...", flush=True)
        X_train, y_train = ema2proba.data_noise_augmentation(nb_aug, 
                                                             X_train, y_train, 
                                                             sigma_noise, 
                                                             isVoicing=True)
        
    if prop_valid > 0:
        print("Splitting test and validation datasets...", flush=True)
        X_test, X_valid, y_test, y_valid = ema2proba.split_data(X_test, y_test, 
                                                      test_size=prop_valid/(prop_test+prop_valid), 
                                                      random_state=42)
    else:
        X_valid = None
        y_valid = None
        
    if model_architecture.lower() == "mlp":
        clf = ema2proba.create_mlp(hidden_layer=hdl, alpha=alpha, 
                                   verbose=verbosity,
                       activation=activation_function, batch_size=batch_size,
                       num_epochs=num_epochs)
        
    else:    
        nb_feat_output = num_class
        nb_epochs = num_epochs
        loss_function = loss_function
        
        clf = ema2proba.build_dnn_classification(nb_feat_input, nb_feat_output, 
                                      hidden_layer_size=hdl,
                                      activation_function=activation_function,
                                      loss_function=loss_function)

    print("Model architecture initialized", flush=True)
    print("Number of training samples: ", len(y_train), flush=True)
    print("Number of testing samples: ", len(y_test), flush=True)
    print("Total number of samples: ", len(y_train) + len(y_test), flush=True)
    print("Number of samples per phoneme:", flush=True)
    for n, c in enumerate(phonemes):
        print("Phone " + c + ": " + "training=" +
               str(y_train.tolist().count(n)) + ", test=" + 
               str(y_test.tolist().count(n)), flush=True)
    print("Number of hidden layer(s): ", len(hdl), flush=True)
    print("Size of hidden layer(s): ", hdl, flush=True)
    print("Activation function: ", activation_function, flush=True)
    print("L2 regularization term: " + "%.2g" %alpha, flush=True)
    
    if model_architecture.lower() == "mlp":
        score = ema2proba.train_mlp(clf, X_train, y_train, X_test, y_test)
    else:
        score = ema2proba.train_encoder(clf, X_train, y_train, X_test, y_test, 
                                        nb_epochs=nb_epochs,
                                        X_valid=X_valid, y_valid=y_valid,
                                        verbosity=verbosity)[1]
        
    print("Training complete", flush=True)
    print("Score of classifier:", score)

    # pickle.dump([clf, scaler, phonemes, score], open(output_file, 'wb'))
    
    return score

delta_vec = range(31)

scores = []
for delta in delta_vec:
    score = training_model(delta)
    scores.append(score)
    
plt.figure()
plt.plot(delta_vec, scores)

