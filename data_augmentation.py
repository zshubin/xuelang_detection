import numpy as np
import random
import Image
from random import randint
from PIL import ImageEnhance
from keras.preprocessing import image as ksimage
from skimage import exposure



NOISE_NUMBER = 5000

#noise
def img_noise(batch):
    s = np.shape(batch)
    for i in range(NOISE_NUMBER):
        x = np.random.randint(0, s[0])
        y = np.random.randint(0, s[1])
        batch[x, y, :] = 255
    return batch

#flip
def flip(batch):
    if bool(random.getrandbits(1)):
        if bool(random.getrandbits(1)):
            batch = img_flip_H(batch)
        else:
            batch = img_flip_V(batch)
    return batch
def img_flip_V(batch):
    vertical_flip_img = batch[::-1, :, :]
    return vertical_flip_img

def img_flip_H(batch):
    horizontal_flip_img = batch[:, ::-1, :]
    return horizontal_flip_img


#rotate_reclock
def ks_rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = ksimage.transform_matrix_offset_center(rotation_matrix, h, w)
    x = ksimage.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def img_rotate0030(batch):
    rotate_limit=(0, 30)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot

def img_rotate3060(batch):
    rotate_limit=(30, 60)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot

def img_rotate6090(batch):
    rotate_limit=(60, 90)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot

def img_rotate90140(batch):
    rotate_limit=(90, 140)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot

def img_rotate140180(batch):
    rotate_limit=(140, 180)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot

def img_rotate4000(batch):
    rotate_limit=(-40, 0)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot

def img_rotate9040(batch):
    rotate_limit=(-90, -40)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot

def img_rotate14090(batch):
    rotate_limit=(-140, -90)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot

def img_rotate180140(batch):
    rotate_limit=(-180, -140)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = ks_rotate(batch, theta)
    return img_rot


def rotate(batch):
    rotateflag = np.random.randint(1, 10)
    if (rotateflag > 4):
        if (rotateflag == 5):
            coll = img_rotate140180(batch)
            return coll
        if (rotateflag == 6):
            coll = img_rotate180140(batch)
            return coll
        if (rotateflag == 7):
            coll = img_rotate14090(batch)
            return coll
        if (rotateflag == 8):
            coll = img_rotate9040(batch)
            return coll
        else:
            coll = img_rotate4000(batch)
            return coll
    else:
        if (rotateflag == 4):
            coll = img_rotate90140(batch)
            return coll
        if (rotateflag == 3):
            coll = img_rotate6090(batch)
            return coll
        if (rotateflag == 1):
            coll = img_rotate0030(batch)
            return coll
        else:
            coll = img_rotate3060(batch)
            return coll

#enhance_contrast
def aj_contrast(batch):
    if bool(random.getrandbits(1)):
        gam = exposure.adjust_gamma(batch, 0.5)
        # log= exposure.adjust_log(image)
        return gam
    else:
        return batch

def randomColor(batch):
    if bool(random.getrandbits(1)):
        random_factor = np.random.randint(0, 31) / 10.
        color_image = ImageEnhance.Color(batch).enhance(random_factor)
        random_factor = np.random.randint(10, 21) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(10, 21) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(0, 31) / 10.
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    else:
        return batch


def color_preprocessing(x_train):
    x_train = x_train.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])
    return x_train

def _random_crop(height, width, image):
    x = randint(0, 2560 - width)
    y = randint(0, 1960 - height)
    image = image[y:y + height, x:x + width]
    return image

def random_crop(image):
    flag = np.random.randint(1, 3)
    if flag == 1:
        image = image
    else:
        image = _random_crop(1680, 2240, image)
    return image

def data_augmentation(batch_input):
    length = len(batch_input)
    batch_output = list()
    for i in range(length):
        batch = np.array(batch_input[i])
        # batch = aj_contrast(batch)
        batch = img_noise(batch)
        batch = flip(batch)
        batch = rotate(batch)
        batch_output.append(batch)
    return batch_output
