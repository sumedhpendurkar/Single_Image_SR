#!/bin/python3

import keras
import numpy as np
import scipy as sp
from keras.optimizers import SGD, Adam

from keras.applications import vgg19
import cv2
import os
from keras.layers import Conv2D, Dense, Deconv2D, MaxPooling2D, UpSampling2D
import sys

def get_model_convolutional():
    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same', strides = (1,1), input_shape=(228, 228, 3)))
    model.add(Conv2D(64, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(Conv2D(128, (3, 3), strides = (1,1), padding = 'same', activation='relu'))
    model.add(Deconv2D(64, (3, 3), strides = (1,1), padding = 'same', activation = 'relu'))
    model.add(Deconv2D(32, (3, 3), strides = (1,1), padding = 'same', activation = 'relu'))
    model.add(Deconv2D(3, (3,3), strides = (1,1), padding = 'same', activation = 'relu'))
    adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-06)
    model.compile(loss='mse', optimizer=adam)
    return model

def get_modified_vgg19():
    model = vgg19.VGG19(weights = 'imagenet', include_top = True)

    for x in range(20):
        model.layers.pop()
    x = UpSampling2D()(model.layers[-1].ouput)
    x = Deconv2D(64, (3,3), padding = 'same', activation = 'relu')(x)
    x = Deconv2D(3, (1,1), padding = 'same', activation = None)(x)
    mod = keras.models.Model(input = model.input, output = x)
    adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-06)
    mod.compile(loss='mse', optimizer = adam)
    return mod

def training_set_conv():
    path_x = './data2/x/'
    path_y = './data2/y/'
    files = os.listdir(path_x)
    max_img_size_x = 228
    max_img_size_y = 228
    train_x = np.zeros((99,max_img_size_x, max_img_size_x, 3), dtype = np.float32)
    train_y = np.zeros((99,max_img_size_y, max_img_size_y, 3), dtype = np.float32)
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

def training_set_vgg():
    path_x = './data3/x/'
    path_y = './data3/y/'
    files = os.listdir(path_x)
    max_img_size_x = 224
    max_img_size_y = 224
    max_img = 99
    train_x = np.zeros((max_img,max_img_size_x, max_img_size_x, 3), dtype = np.float32)
    train_y = np.zeros((max_img,max_img_size_y, max_img_size_y, 3), dtype = np.float32)
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
        print(train_x[i].shape, train_y[i].shape)
    print(train_x.shape)
    print(train_y.shape)
    return vgg19.preprocess_input(train_x), train_y


if __name__ == '__main__':
    model1 = get_modified_vgg19()
    model1.summary()
    x_train, y_train = training_set_vgg()
    print("x_train.size = ", x_train.size)
    print("y_train.size = ", y_train.size)
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    model1.fit(x_train, y_train, verbose = 2,epochs=100, batch_size=16)
    model1.save('sr_deconv_net.h5', overwrite = True)
    y_predicted = model1.predict(x_train[0:5], batch_size = 5)
    for i in range(5):
        cv2.imwrite('predicted_' + str(i)+'.png', np.uint8(y_predicted[i]))
