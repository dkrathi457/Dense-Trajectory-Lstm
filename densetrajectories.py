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


checkpointer = ModelCheckpoint(
        filepath='/home/cis/Desktop/LStm Dense Trajectories/data/checkpoints/' + 'dt'+ \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

# Helper: TensorBoard
tb = TensorBoard(log_dir='/home/cis/Desktop/LStm Dense Trajectories/data/logs')

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger('/home/cis/Desktop/LStm Dense Trajectories/data/logs/'+ '-' + 'training-dt' + \
        str(timestamp) + '.log')


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



model = Sequential()
model.add(LSTM(437, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

optimizer = Adam(lr=1e-6)
model.compile(loss='categorical_crossentropy', optimizer= optimizer,
                           metrics = ['accuracy'] )
#train_x.shape[0]
#train_x_t = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
model.fit(X_train, Y_train, batch_size=32, epochs=70, validation_data=(X_test, Y_test), verbose=1)
