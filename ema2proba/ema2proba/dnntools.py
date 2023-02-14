# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:56:34 2023

@author: belie
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import keras
from keras import layers, models, regularizers

def normalization(X, norm="zscore"):
    norm = "max"
    if norm == 'zscore':    
        scaler = StandardScaler()
    elif norm == 'max':
        scaler = MaxAbsScaler()
    return scaler.fit_transform(X), scaler

def create_mlp(hidden_layer=(100,), alpha=0.1, verbose=1, activation='relu', 
               solver='adam', batch_size=32, num_epochs=200):
    return MLPClassifier(hidden_layer_sizes=hidden_layer, alpha=alpha, verbose=verbose, 
                        activation=activation, solver=solver,
                        batch_size=batch_size, max_iter=num_epochs)
    
def split_data(X, y, test_size=0.1, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state)

def train_mlp(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def train_encoder(clf, X_train, y_train, X_test, y_test, nb_epochs=10,
                  batch_size=32, verbosity=1, X_valid=None, y_valid=None):
    if X_valid is not None and y_valid is not None:
        valid_dataset = (X_valid, y_valid)
    else:
        valid_dataset = (X_test, y_test)
    clf.fit(X_train, y_train,
                    epochs=nb_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=valid_dataset,
                    verbose=verbosity)
    return clf.evaluate(valid_dataset[0], valid_dataset[1])
    
def confumtx(clf, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    
def build_dnn_classification(nb_feat_input, nb_feat_output, 
                    hidden_layer_size=(40, 80, 40), 
                    activation_function="relu",
                    loss_function="mean_squared_error"):
    
    encoding_function = activation_function
    
    input_art = keras.Input(shape=(nb_feat_input,))
    nb_layers = len(hidden_layer_size)
    
    hidden_layers = [layers.Dense(hidden_layer_size[0], 
                                activation=activation_function)(input_art)]
    for n in range(1, nb_layers):
        hidden_layers.append(layers.Dense(hidden_layer_size[n], 
                                activation=encoding_function)(hidden_layers[n-1]))
    decoded = layers.Dense(nb_feat_output, 
                            activation="softmax")(hidden_layers[-1])
    encoder = keras.Model(input_art, decoded)
    
    encoder.compile(optimizer='adam', loss=loss_function, metrics=["accuracy"])
    return encoder