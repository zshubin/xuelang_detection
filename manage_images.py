# coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data_augmentation import *
import numpy as np
import os
from PIL import Image

path = './data/train/N/'

avg = np.array([[[158.14044652223768, 166.1790830525745, 176.01642383807538]]])
def load_data():
    x_train = np.empty((1407, 224, 224, 3), dtype='float32')
    y_train = np.empty((1407,), dtype='uint8')
    train_imgsN = os.listdir('./data/train/N/')
    train_imgsY = os.listdir('./data/train/Y/')
    count1 = len(train_imgsN)
    count2 = len(train_imgsY)
    # labels = []
    for j in range(count1):
        img_train = Image.open('data/train/N/' + train_imgsN[j])
        # img_train = randomColor(img_train)
        img_train = img_train.resize((224, 224), Image.ANTIALIAS)
        arr_train = np.asarray(img_train, dtype='float32')
        x_train[j, :, :, :] = arr_train - avg
        y_train[j] = int(0)
    for j in range(count1, count1+count2):
        img_train = Image.open('data/train/Y/'+train_imgsY[j-count1])
        # img_train = randomColor(img_train)
        img_train = img_train.resize((224, 224), Image.ANTIALIAS)
        arr_train = np.asarray(img_train, dtype='float32')
        x_train[j, :, :, :] = arr_train
        y_train[j] = int(1)
    y_train = np.array([[float(i == label) for i in range(2)] for label in y_train])
    permutation_train = np.random.permutation(x_train.shape[0])
    X_train = x_train[permutation_train, :, :, :] - avg
    Y_train = y_train[permutation_train]
    return X_train, Y_train


def load_data_test():
    # p = np.empty((1, 224, 224, 3), dtype='float32')
    test_imgs = os.listdir('./data/train/Y/')
    count = len(test_imgs)
    img_name = []
    img_value = np.empty((892, 224, 224, 3), dtype='float32')
    for j in range(count):
        name = str(test_imgs[j])
        img_test = Image.open('./data/train/Y/'+name)
        img_test = img_test.resize((224, 224), Image.ANTIALIAS)
        img_test = np.asarray(img_test, dtype='float32')
        img_value[j, :, :, :] = img_test - avg
        img_name.append(name)
    return img_name, img_value
