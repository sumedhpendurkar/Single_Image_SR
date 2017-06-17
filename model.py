#!/bin/python3

import keras
import numpy as np
import scipy as sp
from keras.optimizers import SGD

import cv2
import os
from keras.layers import Conv2D, Dense

def get_model_convolutional():
    model = keras.models.Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', strides = (1,1), input_shape=(100, 100, 3)))
    model.add(Conv2D(3, (3, 3), strides = (1,1), activation=None)) 
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def get_model_ann():
    model = keras.models.Sequential()
    model.add(Dense(128, activation = 'relu', input_dim = 100 * 100 * 3))
    model.add(Dense(100*3*3, activation = None))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def prepare_training_inputs():
    path = './dataset/x/'
    x = os.listdir(path)
    train_x = []
    tmp = [0 for x in range(1024)]
    for i in x:
        print(i)
        img = cv2.imread(path + i)
        #np.pad(img,((0, 1024 - img.shape[0]), (0,1024 - img.shape[1])), mode = 'constant', constant_values = 0)
        a = img.tolist()
        for x in range(len(a)):
            while len(a[x]) !=1024:
                a[x].append(0)
        while len(a) !=1024:
            a.append(tmp)
        train_x.append(a)
        print(np.array(a).shape)
    return np.array(train_x)

def prepare_training_outputs():
    pass

if __name__ == '__main__':
    model1 = get_model_convolutional()
    #model2 = get_model_ann()
    x_train = prepare_training_inputs()
    y_train = prepare_training_outputs()
    model1.fit(x_train, y_train, epochs=20, batch_size=128)

