# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
from os.path import join
import tensorflow as tf

from PIL import Image
import cv2

# TFRcord文件
SPECIFY_TRAIN_FILE = 'specify_train.tfrecords'

TRAIN_FILE = 'train.tfrecords'

# 图片信息

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS
LABEL_PIXELS=IMG_HEIGHT * IMG_WIDTH * 21



def read_and_decode(filename_queue):
    # 创建一个reader来读取TFRecord文件中的样例
    tf_record_filename_queue = tf.train.string_input_producer([filename_queue])
    reader = tf.TFRecordReader()
    # 从文件中读出一个样例
    #_, serialized_example = reader.read(filename_queue)
    _,serialized_example = reader.read(tf_record_filename_queue)
    # 解析读入的一个样例
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.string),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'image_name': tf.FixedLenFeature([], tf.int64)

    })
    # 将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['image_raw'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    image_name = tf.cast(features['image_name'], tf.int64)

    image.set_shape([IMG_PIXELS])
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    image = tf.cast(image, tf.float32)

    label.set_shape([LABEL_PIXELS])
    label = tf.reshape(label, [IMG_HEIGHT, IMG_WIDTH,21])
    label = tf.cast(label, tf.float32)


    return image, label,image_name


# 用于获取一个batch_size的图像和label
def inputs(data_set, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None
    if data_set == 'train':
        file = TRAIN_FILE
    elif data_set=='specifytrain':
        file = SPECIFY_TRAIN_FILE
    print('train_dataset',file)
    #with tf.name_scope('input') as scope:
        #filename_queue = tf.train.string_input_producer([file], num_epochs=num_epochs)
    image, label,image_name = read_and_decode(file)
    # 随机获得batch_size大小的图像和label
    #images, labels = tf.train.shuffle_batch([image, label],
     #                                       batch_size=batch_size,
      #                                      num_threads=64,
       #                                     capacity=1000 + 3 * batch_size,
        #                                    min_after_dequeue=1000
         #                                   )
    b_images,b_labels,b_imagename = tf.train.batch([image, label,image_name],
                                            batch_size=batch_size
                                            )
    #return images, labels
    return b_images,b_labels,b_imagename


def main(argv):
    #filename_queue = tf.train.string_input_producer([TRAIN_FILE])
    image, label = read_and_decode(TRAIN_FILE)
    print(label)
    sess = tf.Session()
    
    
    print(sess.run(label))
    # images,labels=inputs('train',100,2)
    # print(images)
    # print(labels)


if __name__ == '__main__':
    tf.app.run()
