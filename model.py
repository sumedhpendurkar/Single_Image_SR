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
    model.add(Conv2D(128, (3, 3), activation='relu', strides = (1,1), input_shape=(1028, 1028, 3)))
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

def training_x():
    path_x = './dataset/x/'
    path_y = './dataset/y/'
    files = os.listdir(path_x)
    max_img_size_x = 1028
    max_img_size_y = 1024
    train_x = np.zeros((108,max_img_size_x, max_img_size_x, 3), dtype = np.uint8)
    train_y = np.zeros((108,max_img_size_y, max_img_size_y, 3), dtype = np.uint8)

    for i in range(0, len(files)):
        img =  cv2.imread(path_x+files[i])
        print(files[i])
        for j in range(len(img)):
            for k in range(len(img[j])):
                train_x[i][j][k][:] = img[j][k]
        file_y = files[i].replace('bicubic', 'HR')
        img = cv2.imread(path_y+file_y)
        print(file_y)
        for j in range(len(img)):
            for k in range(len(img[j])):
                train_y[i][j][k][:] = img[j][k]
    print(train_x.shape)
    print(train_y.shape)
    return train_x, train_y

if __name__ == '__main__':
    model1 = get_model_convolutional()
    #model2 = get_model_ann()
    model1.summary()
    x_train, y_train = training_x()
    print("x_train.size = ", x_train.size)
    print("y_train.size = ", y_train.size)
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    model1.fit(x_train, y_train, verbose = 2,epochs=20, batch_size=4)

