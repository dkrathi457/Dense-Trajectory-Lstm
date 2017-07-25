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
import random
import os

filepath = "/home/cis/Desktop/LStm Dense Trajectories/training.txt"
training_data_list = pd.read_csv(filepath, sep=" ", header=None)


filepath_test = "/home/cis/Desktop/LStm Dense Trajectories/testingdata.txt"
testing_data_list = pd.read_csv(filepath_test, sep=" ", header=None)

test_train_list = pd.concat([training_data_list, testing_data_list])

classes = ['boxing' , 'handclapping' , 'handwaving' , 'jogging' , 'running' , 'walking']


## Get label from file name and convert to categorical
    
def make_label_data(file_name, classes):
    label = file_name.split("_")
    label = label[1]
    label_encoded = classes.index(label)
    get_label = np_utils.to_categorical (label_encoded, len(classes))       
    get_label  = get_label[0]
    return get_label


def get_size(alldata):
    seq_size = []
    for i in range(0, len(alldata)):
        filename = alldata[i][0]
        #print filename
        filepath_file = os.path.join("/home/cis/Desktop/LStm Dense Trajectories/Dense Trajectories" , filename)
        x= pd.read_csv(filepath_file, sep = "\t", header =None)
        size = len(x)
        seq_size.append(size)
    return seq_size


def get_min_value(test_train_list):
    print "Getting Sequence size"
    seq_size = get_size(test_train_list)
    print "Getting Minimum value"
    minvalue = seq_size[0]
    
    for i in range(0, len(seq_size)):
        if seq_size[i] < minvalue:
            minvalue = seq_size[i]
        else:
            minvalue = minvalue
    return minvalue

#seq_size = get_size(test_train_list.values)


## Get the training data and labels for the file

def load_training_data(training_data_list, classes, minvalue):
    X, Y = [],[]
    for i in range(0, len(training_data_list)):
        filename = training_data_list.iloc[i][0]
        print filename
        filepath_file = os.path.join("/home/cis/Desktop/LStm Dense Trajectories/Dense Trajectories" , filename)
        x= pd.read_csv(filepath_file, sep = "\t", header =None) 
        x_sample = x.iloc[random.sample(x.index, minvalue)] 
        x = x_sample.values
        label = make_label_data(filename, classes)
        X.append(x)
        Y.append(label)
        ## get label
    return np.array(X), np.array(Y)    

## Load the testing data and labels

def load_testing_data(testing_data_list, classes, minvalue):
    X, Y = [],[]
    for i in range(0, len(testing_data_list)):
        filename = testing_data_list.iloc[i][0]
        print filename
        filepath_file = os.path.join("/home/cis/Desktop/LStm Dense Trajectories/Dense Trajectories" , filename)
        x= pd.read_csv(filepath_file, sep = "\t", header =None) 
        x_sample = x.iloc[random.sample(x.index, minvalue)] 
        x = x_sample.values
        label = make_label_data(filename, classes)
        X.append(x)
        Y.append(label)
        ## get label
    return np.array(X), np.array(Y)    

minvalue = get_min_value(test_train_list.values)

X_train, Y_train = load_training_data(training_data_list, classes, minvalue)

X_test , Y_test = load_testing_data(testing_data_list, classes, minvalue)

