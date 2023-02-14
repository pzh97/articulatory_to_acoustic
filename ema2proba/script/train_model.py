# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:53:48 2023

@author: belie
"""

from os import path, environ
import ema2proba
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import argparse
import time
plt.close("all")

start = time.time()

parser = argparse.ArgumentParser(description=""""Training articulatory 
                                 to probability model""")
parser.add_argument("data_file", metavar="data_file", type=str,
                    help="""Path to file containing data (or file containing
                    containing a list of data files)""")
parser.add_argument("output_file", metavar="output_file", type=str, 
                    help="File where to save the model")
parser.add_argument("-us", "--undersampling", dest="undersampling",
                    metavar="undersampling", type=int,
                    help="""Theshold above which a class will be undersampled 
                    (default=0, no undersampling)""", 
                    default=0)
parser.add_argument("-os", "--oversampling", dest="oversampling",
                    metavar="oversampling", type=int,
                    help=""""Decision to ovsersample or not 
                    (default=0, no oversampling)""", 
                    default=0)
parser.add_argument("-phm", "--phoneme-merging", dest="phoneme_merging", 
                    metavar="phone_merging", nargs="+", type=str,
                    help="""Phonemes to merge. Should be a sequence of strings as 
                    a-b c-d ..., where b is merged with a and d is merged with c
                    (default=None)""", 
                    default=None)
parser.add_argument("-phd", "--phoneme-mdeleting", dest="phoneme_deleting", 
                    metavar="phoneme_deleting", nargs="+", type=str,
                    help="""Phonemes to delete. Should be a sequence of strings as 
                    a b, where a nd b will be deleted (default=None)""", 
                    default=None)
parser.add_argument('-ls', dest='layer_size',
                    metavar='hidden layer size', type=int, nargs='+',
                    help="""size of the hidden layers (default=100)""", 
                    default=(100,))
parser.add_argument('-lf', dest='loss_function',
                    metavar='loss_function', type=str, 
                    help="""Loss function (default="sparse_categorical_crossentropy")""", 
                    default="sparse_categorical_crossentropy")
parser.add_argument('-a', '--act', dest='activation_function',
                    metavar='activation function', type=str,
                    help='activation function (default=tanh)', 
                    default="tanh")
parser.add_argument('-ne', dest='num_epochs', type=int,
                    metavar='number of epochs',
                    help='number of epochs (default=15)', default=15)
parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, 
                    metavar='batch size',
                    help='batch size (default=32)', 
                    default=32)
parser.add_argument('-model', '--model', dest='model', type=str, 
                    metavar="model_architecture",
                    help="""Choice of the global architecture. 
                    Should be either DNN or MLP (default=MLP)""",
                    default="MLP")
parser.add_argument('-v', '--verbosity', dest='verbosity', type=int, 
                    metavar='verbosity',
                    help='verbosity level (default=1)', default=1)
parser.add_argument('-p', '--test', dest='proportional_test',
                    type=float, metavar='percentage',
                    help='percentage of data for test (default=5%)', 
                    default=5)
parser.add_argument('-pv', '--valid', dest='proportional_validation',
                    type=float, metavar='percentage_validation',
                    help='percentage of data for validation (default=5%)', 
                    default=5)
parser.add_argument('-d', '--delta', dest='delta',
                    type=int, metavar='delta',
                    help="""Number of previous and next samples 
                    to add as features (default=5)""", 
                    default=5)
parser.add_argument('-da', '--data-augmentation', dest='data_augmentation',
                    type=int, metavar='data_augmentation',
                    help="""Number of new sets for
                    noise-based data augmentation (default=0, no DA)""", 
                    default=0)
parser.add_argument('-s', '--sigma', dest='sigma',
                    type=float, metavar='sigma',
                    help="""Standard deviation of additional Gaussian noise for 
                    data augmentation (default=0.1)""", 
                    default=0.1)
parser.add_argument('-al', '--alpha', dest='alpha', type=float,
                    metavar='alpha',
                    help="""L2 regularization term for MLP (default=0)""", 
                    default=0)
parser.add_argument('-vm', '--voicing_merged', dest='voicing_merging', 
                    type=int, metavar='voicing_merging',
                    help="""Decision to keep voicing (default=0, keep voicing)""", 
                    default=0)
parser.add_argument('-n', '--normalization', dest='normalization', 
                    type=str, metavar='normalization',
                    help="""Style of normalization (default=zscore)""", 
                    default="zscore")

data_file = parser.parse_args().data_file
output_file = parser.parse_args().output_file
undersampling = parser.parse_args().undersampling
oversampling = parser.parse_args().oversampling
phoneme_merging = parser.parse_args().phoneme_merging
phoneme_delete = parser.parse_args().phoneme_deleting
hdl = parser.parse_args().layer_size
activation_function = parser.parse_args().activation_function
num_epochs = parser.parse_args().num_epochs
batch_size = parser.parse_args().batch_size
model_architecture = parser.parse_args().model
verbosity = parser.parse_args().verbosity
prop_test = parser.parse_args().proportional_test
prop_valid = parser.parse_args().proportional_validation
delta = parser.parse_args().delta
data_augmentation = parser.parse_args().data_augmentation
sigma_noise = parser.parse_args().sigma
alpha = parser.parse_args().alpha
loss_function = parser.parse_args().loss_function
voicing_merging = parser.parse_args().voicing_merging
normalization = parser.parse_args().normalization

if verbosity > 0:
    print("Start script", flush=True)
    print("Input parameters:", flush=True)
    print("Data contained in file:", data_file, flush=True)
    if phoneme_merging is not None:
        print("Merging phonemes:", phoneme_merging, flush=True)
    else:
        print("No phoneme to merge", flush=True)
    if phoneme_delete is not None:
        print("Phoneme(s) to delete:", phoneme_delete, flush=True)
    else:
        print("No phoneme to delete", flush=True)
    if undersampling > 0:
        print("Undersampling threshold:", undersampling, flush=True)
    else:
        print("No undersampling", flush=True)
    if oversampling == 1:
        print("Classes will be oversampled", flush=True)
    else:
        print("No oversampling", flush=True)
    print("Standardization:", normalization, flush=True)
    print("Model architecture:", model_architecture, flush=True)
    print("Number of hidden layers:", len(hdl), flush=True)
    print("Sizes of hidden layers:", hdl, flush=True)
    print("Activation function:", activation_function, flush=True)
    if model_architecture.lower() == "mlp":
        print("L2 regularization term:", alpha, flush=True)
    else:
        print("Loss function:", loss_function, flush=True)
    print("Number of epochs:", num_epochs, flush=True)
    print("Batch size:", batch_size, flush=True)
    print("Number of epochs:", num_epochs, flush=True)
    if delta > 0:
        print("Number of delta:", delta, flush=True)
    else:
        print("No delta")
    if data_augmentation > 0:
        print("Number of augmented data set (Gaussian noise):",
              data_augmentation, flush=True)
        print("Added noise standard deviation:", sigma_noise, flush=True)
    else:
        print("No data augmentation")
    print("Proportion of test data: " + str(prop_test) + "%", flush=True)
    if prop_valid > 0:
        print("Proportion of validation data: " + str(prop_valid) + "%", flush=True)
    else:
        print("No validation data set")
    if voicing_merging > 0:
        print("Voiced/unvoiced phonemes are merged. Voicing feature is removed", flush=True)
    else:
        print("No Voiced/unvoiced merging", flush=True)
    
    print("Results will be stored in:", output_file, flush=True)

if not path.isfile(data_file):
    raise ValueError("Data file " + data_file + " does not exist!")
else:
    with h5py.File(data_file, 'r') as hf:
        features = hf["features"][()]
        labels = hf["labels"][()]
        phonemes = [(p[0]).decode("ascii") for p in hf["phonemes"][()]]
    
if phoneme_merging is not None:
    labels, phonemes = ema2proba.merge_phoneme(phoneme_merging, phonemes, 
                                               labels, verbosity=verbosity)
if voicing_merging:
    print("Voicing merging...")
    phoneme_merging = "P-B T-D K-G SH-ZH S-Z TH-DH F-V".split(" ")
    labels, phonemes = ema2proba.merge_phoneme(phoneme_merging, phonemes, 
                                               labels, verbosity=verbosity)
    features = features[:, :-1]
    
X, y = (features, labels.reshape(-1))     

if undersampling > 0:
    print("Performing undersampling...")
    X, y = ema2proba.undersampling(X, y, threshold=undersampling)

cl = np.unique(y)
nb_cl = [np.count_nonzero(y == c) for c in cl]
num_class = len(cl)

if delta > 0:
    X, y = ema2proba.add_delta_features(X, delta, y)

print("Normalization...", flush=True)
X, scaler = ema2proba.normalization(X, norm=normalization)
print("Splitting train and test datasets...", flush=True)
X_train, X_test, y_train, y_test = ema2proba.split_data(X, y, 
                                              test_size=(prop_test+prop_valid)/100, 
                                              random_state=42)

nb_aug = data_augmentation
nb_sampl, nb_feat_input = X_train.shape
if nb_aug > 0:
    print("Performing data augmentation on training dataset...", flush=True)
    for n in tqdm(range(nb_aug), desc="Data augmentation"):
        nb_sampl, nb_feat_input = X_train.shape
        Xaugment = np.hstack((X_train[:,:-1]*(1 + sigma_noise * np.random.randn(nb_sampl, 
                                                                        nb_feat_input-1)),
                              X_train[:, -1].reshape(-1, 1)))
        X_train = np.vstack((X_train, Xaugment))
        y_train = np.vstack((y_train.reshape(-1, 1), y_train.reshape(-1, 1))).reshape(-1)

if prop_valid > 0:
    print("Splitting test and validation datasets...", flush=True)
    X_test, X_valid, y_test, y_valid = ema2proba.split_data(X_test, y_test, 
                                                  test_size=prop_valid/(prop_test+prop_valid), 
                                                  random_state=42)
else:
    X_valid = None
    y_valid = None
    
if model_architecture.lower() == "mlp":
    clf = ema2proba.create_mlp(hidden_layer=hdl, alpha=alpha, verbose=1,
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

# clf.fit(X_train, y_train)
# score = clf.score(X_test, y_test)
if model_architecture.lower() == "mlp":
    score = ema2proba.train_mlp(clf, X_train, y_train, X_test, y_test)
else:
    score = ema2proba.train_encoder(clf, X_train, y_train, X_test, y_test, 
                                    nb_epochs=nb_epochs,
                                    X_valid=X_valid, y_valid=y_valid)[1]
print("Training complete", flush=True)
print("Score of classifier:", score)

import pickle
pickle.dump([clf, scaler, phonemes, score], open(output_file, 'wb'))



