from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import manage_images
import numpy as np
IMG_CLASSES = ['N', 'Y']

DATA_DIR = 'data/'
TRAIN_DATA_PATH = 'data/train/'
TEST_DATA_PATH = 'data/test/'

IMG_HEIGHT = int(224)
IMG_WIDTH = int(224)
IMG_CHANNELS = 3
NUM_FILES_DATASET = 1407 
NUM_TRAIN_EXAMPLES = 1047
NUM_KAGGLE_TEST = 12



def pre_processing(data_set='train',batch_size=32):
    if data_set is 'train':
        images, labels = manage_images.load_data()
        # train_mean = np.mean(images, axis=0)
        train_images = []
        train_labels = []
        for i in range(1407):
            train_images.append(images[i])
            train_labels.append(labels[i])
        batch_train_set = {'images': train_images, 'labels': train_labels}
        image_num = {'train': NUM_TRAIN_EXAMPLES}
        return batch_train_set, image_num
    elif data_set is 'test':
        image_name, image_value = manage_images.load_data_test()
        # train_mean = np.mean(images, axis=0)
        test_images = []
        for i in range(len(image_name)):
            test_images.append(image_value[i])
        batch_test_set = {'images': test_images, 'names': image_name}
        return batch_test_set



