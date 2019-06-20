import os
import numpy as np
from data_augmentation import random_crop
import cv2
from keras.preprocessing import image

classes = ['N', 'Y']
gen_data = image.ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True)
class DataLoader(object):
    def __init__(self):
        self.classes = classes
        self.images_urls, self.labels = self.load_all_data()
        self.cursor = 0
        self.avg = np.array([[[158.14044652223768, 166.1790830525745, 176.01642383807538]]])



    def load_all_data(self, type='train'):
        img_path_urls = []
        label = []
        for i in range(len(classes)):
            images = os.listdir('./data/{}/{}/'.format(type, classes[i]))
            images = ['./data/{}/{}/'.format(type, classes[i])
                                  + image for image in images]
            img_path_urls.extend(images)
            label.extend([i]*len(images))
            print('{}: {} has {} images'.format(type, classes[i], len(images)), '===', i, classes[i])
        print('\ntotal images num: {}'.format(len(img_path_urls)))
        img_path_urls = np.array(img_path_urls)
        label = np.array(label)
        indices = np.random.permutation(len(img_path_urls))
        return img_path_urls[indices], label[indices]

    def get_batch_data(self, batch_size):
        images = np.zeros([batch_size, 224, 224, 3])
        labels = np.zeros([batch_size, len(self.classes)])
        for i in range(batch_size):
            if self.cursor < self.images_urls.shape[0]:
                images[i, :] = (self.get_image(self.images_urls[self.cursor]))
                labels[i, :] = self.one_hot(self.labels[self.cursor], len(classes))
                self.cursor += 1
        return images, labels


    def shuffle(self):
        perm = np.arange(len(self.images_urls))
        np.random.shuffle(perm)
        self.images_urls = self.images_urls[perm]
        self.labels = self.labels[perm]
        self.cursor = 0

    def get_image(self, image_url):
        img = cv2.imread(image_url)
        img = np.reshape(img, [1, 1920, 2560, 3])
        for images in gen_data.flow(img, batch_size=1, shuffle=False):
            img = images
            break
        img = np.reshape(img, [1920, 2560, 3])
        img = random_crop(img)
        img = cv2.resize(img, (224, 224))
        return img

    def one_hot(self, i, num_classes):
        label = np.zeros((1, num_classes))
        label[0, i] = 1
        return label

    def get_test_image(self, url):
        if os.path.exists(url):
            img = cv2.imread(url)
            return self.do_change(img)
        else:
            return None

