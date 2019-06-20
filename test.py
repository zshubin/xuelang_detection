from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data_processing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import csv as csv
import numpy as np

CHECK_POINT_PATH = 'check_point/train_model.ckpt'
NUM_KAGGLE_TEST = 12500
BATCH_SIZE = 50
batch_test_set = data_processing.pre_processing(data_set='test', batch_size=BATCH_SIZE)

prediction_file = open('result_file.csv', 'wb')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(['filename', 'probability'])

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    keep_prob = tf.placeholder(tf.float32)
    logits, _ = nets.resnet_v2.resnet_v2_50(inputs=images, num_classes=2, is_training=False)


    # pros = tf.nn.softmax(logits)
    restorer = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        restorer.restore(sess, CHECK_POINT_PATH)
        value = np.empty((1, 224, 224, 3))
        for i in range(892):
            proby = np.zeros([1, 2])
            name = batch_test_set['names'][i]
            value[0, :, :, :] = batch_test_set['images'][i]
            image_pros = sess.run(logits, feed_dict={images: value, keep_prob: 1.0})
            image_pros = tf.reshape(image_pros, [1, 2])
            proby = image_pros.eval()
            temp = 0
            if proby[0, 0] > proby[0, 1]:
               temp = 0.995
            else:
               temp = 0.005                 
            prediction_file_object.writerow([name, temp])
prediction_file.close()
