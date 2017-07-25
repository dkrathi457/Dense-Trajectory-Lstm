#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:44:13 2017

@author: cis
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os.path
from keras.utils import np_utils
import tensorflow as tf
from pandas import Series
import time
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam


checkpointer = ModelCheckpoint(
        filepath='/home/cis/Desktop/LStm Dense Trajectories/data/checkpoints/' + 'bow'+ \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

# Helper: TensorBoard
tb = TensorBoard(log_dir='/home/cis/Desktop/LStm Dense Trajectories/data/logs')

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger('/home/cis/Desktop/LStm Dense Trajectories/data/logs/'+ '-' + 'training-' + \
        str(timestamp) + '.log')




trainingdict = loadmat('/home/cis/Videos/DSC550/SVM/All desc/dictan_Alldesc1.mat')
train_x = trainingdict['encode']
train_y = trainingdict['label']


testing_dict = loadmat('/home/cis/Videos/DSC550/SVM/All desc/dictan_Alldesc1_t.mat')
test_x = testing_dict['encode_t']
test_y = testing_dict['label_t']

classes = ['boxing' , 'handclapping' , 'handwaving' , 'jogging' , 'running' , 'walking']


## Convert label to categorical
def convert_to_categ(train_y):
    y = []
    for i in range(0 , len(train_y)):
        k= train_y[i][0]
        #print k
        label_encoded = classes.index(k)
        get_label = np_utils.to_categorical (label_encoded, len(classes))       
        get_label  = get_label[0]
        y.append(get_label)
        
    return np.array(y)

x =[]

train_x_t = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
test_x_t = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])    
 
## Convert label to categorical

def convert_to_categ_test(test_y):
    y = []
    for i in range(0 , len(test_y)):
        k= test_y[i][0]
        #print k
        label_encoded = classes.index(k)
        get_label = np_utils.to_categorical (label_encoded, len(classes))       
        get_label  = get_label[0]
        y.append(get_label)
        
    return np.array(y)

train_y = convert_to_categ(train_y)
test_y = convert_to_categ_test(test_y)

#X = np.array(train_x)    

model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(train_x_t.shape[1], train_x_t.shape[2]),dropout=0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

optimizer = Adam(lr=1e-6)
model.compile(loss='categorical_crossentropy', optimizer= optimizer,
                           metrics = ['accuracy'])
#train_x.shape[0]
#train_x_t = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
model.fit(train_x_t, train_y, batch_size=32, epochs=70, validation_data=(test_x_t, test_y), verbose=1, callbacks=[checkpointer, tb, early_stopper, csv_logger])