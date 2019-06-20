from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from data_augmentation import *
import data_processing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import os
import os.path
import time
TRAIN_LOG_DIR = os.path.join('Log/train/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
TRAIN_CHECK_POINT = 'check_point/train_model.ckpt'
VALIDATION_LOG_DIR = 'Log/validation/'
BATCH_SIZE = 32
EPOCH = 20
if not tf.gfile.Exists(TRAIN_LOG_DIR):
    tf.gfile.MakeDirs(TRAIN_LOG_DIR)

if not tf.gfile.Exists(VALIDATION_LOG_DIR):
    tf.gfile.MakeDirs(VALIDATION_LOG_DIR)





def get_accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.float32, [None, len(data_processing.IMG_CLASSES)])
    keep_prob = tf.placeholder(tf.float32)
    logits, _ = nets.resnet_v2.resnet_v2_50(inputs=images, num_classes=2,  is_training=True)
    logits = tf.reshape(logits, [-1, 2])
    with tf.name_scope('cross_entropy'):
         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    tf.summary.scalar('cross_entropy', loss)
    learning_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.name_scope('accuracy'):
         accuracy = get_accuracy(logits, labels)
    tf.summary.scalar('accuracy', accuracy)
    
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(TRAIN_LOG_DIR)
        ckpt = tf.train.get_checkpoint_state('./check_point')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        step = 0
        for ep in range(EPOCH):
            all_accuracy = 0
            all_loss = 0
            pre_index = 0
            cou = 0
            batch_train_set, images_num = data_processing.pre_processing(data_set='train', batch_size=BATCH_SIZE)
            for i in range(44):
                true_num = 0
                if pre_index +BATCH_SIZE < 1407:
                    feed_image = batch_train_set['images'][pre_index:pre_index + BATCH_SIZE]
                    feed_label = batch_train_set['labels'][pre_index:pre_index + BATCH_SIZE]
                else:
                    feed_image = batch_train_set['images'][pre_index:]
                    feed_label = batch_train_set['labels'][pre_index:]
                feed_image = data_augmentation(feed_image)
                _, accuracy_out, loss_out, summary, out = sess.run([optimizer, accuracy, loss, merged, logits],
                                       feed_dict={images: feed_image, \
                                                  labels: feed_label,
                                                  keep_prob: 0.5})
                for j in range(BATCH_SIZE):
                    if np.argmax(out[j, :]) == np.argmax([feed_label[j, :]]):
                        cou += 1
                        true_num += 1
                train_writer.add_summary(summary, step)
                pre_index += BATCH_SIZE
                step += 1
                all_accuracy += accuracy_out
                all_loss += loss_out
                if i % 10 == 0:
                    print("Epoch %d: Batch %d accuracy is %.2f; Batch loss is %.5f,true_num is %2d/%2d" %(ep + 1, i, accuracy_out, loss_out, true_num, BATCH_SIZE))
                    

            print("Epoch %d: Train accuracy is %.2f; Train loss is %.5f,true_num is %2d/%2d" %(ep + 1, all_accuracy / 44 , all_loss / 44, cou, 1407))
            saver.save(sess, TRAIN_CHECK_POINT)
