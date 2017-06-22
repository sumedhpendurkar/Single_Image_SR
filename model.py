#!/bin/python3

import keras
import numpy as np
import scipy as sp
from keras.optimizers import SGD

import cv2
import os
from keras.layers import Conv2D, Dense, Deconv2D
import sys

def get_model_convolutional():
    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', strides = (1,1), input_shape=(1028, 1028, 3)))
    model.add(Conv2D(64, (3, 3), strides = (1,1), activation='relu'))
    model.add(Deconv2D(32, (3, 3), strides = (1,1), activation = None))
    model.add(Deconv2D(3, (3,3), strides = (1,1), activation = None))
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    return model

def training_set():
    path_x = './dataset/x/'
    path_y = './dataset/y/'
    files = os.listdir(path_x)
    max_img_size_x = 1024
    max_img_size_y = 1024
    train_x = np.zeros((108,max_img_size_x, max_img_size_x, 3), dtype = np.float32)
    train_y = np.zeros((108,max_img_size_y, max_img_size_y, 3), dtype = np.float32)

    for i in range(0, len(files)):
        img =  cv2.imread(path_x+files[i])
        print(files[i])
        for j in range(len(img)):
            for k in range(len(img[j])):
                train_x[i][j][k][:] = img[j][k]
        print(train_x[i])
        file_y = files[i].replace('bicubic', 'HR')
        img = cv2.imread(path_y+file_y)
        print(file_y)
        for j in range(len(img)):
            for k in range(len(img[j])):
                train_y[i][j][k][:] = img[j][k]
        print(train_x[i].shape, train_y[i].shape)
        print(train_y[i])
    print(train_x.shape)
    print(train_y.shape)
    return train_x, train_y

if __name__ == '__main__':
    model1 = get_model_convolutional()
    #model2 = get_model_ann()
    model1.summary()
    x_train, y_train = training_set()
    print("x_train.size = ", x_train.size)
    print("y_train.size = ", y_train.size)
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    model1.fit(x_train, y_train, verbose = 2,epochs=10, batch_size=4)
    model1.save('sr_deconv_net.h5')
    y_predicted = model1.predict(x_train[0:5], batch_size = 5)
    for i in range(5):
        cv2.imwrite('predicted_' + str(i)+'.png', np.uint8(y_predicted[i]))
