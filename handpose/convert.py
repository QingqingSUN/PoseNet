# -*- coding: utf-8 -*-



from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import time
import scipy.io as sio
from scipy.misc import imread, imresize
from os import walk
from os.path import join

# 指定手势的标注文件位置
SPECIFY_LABELPATH = 'E:/python/ServerCode/Data/anno_hand/anno_hand.mat'
# 指定手势图片存放位置

SPECIFY_DATA_DIR = 'E:/python/ServerCode/Data/anno_hand/test/'

# # 标注文件位置
# LABELPATH = '/Data/tensorflow/data/CHD/train/anno_training.mat'
# # 图片存放位置
#
# DATA_DIR = '/Data/tensorflow/data/CHD/train/color/'

# 标注文件位置
LABELPATH = 'E:/python/ServerCode/Data//CHD/train/anno_training.mat'
# 图片存放位置

DATA_DIR = 'E:/python/ServerCode/Data//CHD/train/color/'

# 图片信息
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 读取图片
def readToTfrecord(path):

    filenames = next(walk(path))[2]
    num_files = len(filenames)
    keypoint_hw21=tf.placeholder(tf.float32,shape=(21,2))
    scoremap_size=([IMG_WIDTH,IMG_HEIGHT])
    sigma=tf.constant(25)
    scoremap = create_multiple_gaussian_map(keypoint_hw21, scoremap_size, sigma, None)

    # 加载标注文件
    anno = sio.loadmat(LABELPATH)
    # 加载指定手势标注文件
    # anno = sio.loadmat(SPECIFY_LABELPATH)

    # 遍历所有的图片和label，将图片resize到[256,256,3]  label是[21,2]shape的array
    # score=list()

    record_name = 'train.tfrecords'
    # record_name = 'specify_train.tfrecords'
    writer = tf.python_io.TFRecordWriter(record_name)
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    tf.train.start_queue_runners(sess=sess)
    sess.run(tf.global_variables_initializer())
    
    for i, filename in enumerate(filenames):

        img = imread(join(path, filename))
        img = imresize(img, (IMG_HEIGHT, IMG_WIDTH))
        img=img/255-0.5
        # img.dtype=np.float32

        images = img.astype(np.float32)

        # print('images shape',images,images.dtype)

        # frame=Frame(fra,i)
        keypoint_hw = anno['frame' + (str)(i)]

        stub=filename.split('.')

        image_index=(int)(stub[0])

        keypoint_hw = anno['frame' + (str)(image_index)]



        start=time.time()
        label=sess.run(scoremap,feed_dict={keypoint_hw21:keypoint_hw})
        #label=sess.run(scoremap)
        duration_sess=time.time()-start

        print('cost %d sec'% duration_sess)

        images = images.tostring()
        label = label.tostring()
        # print('label',len(label))
        imagename=(int)(image_index)
        # 将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _bytes_feature(label),
            'image_raw': _bytes_feature(images),
            'image_name':_int64_feature(imagename)}))
        # 将example写入TFRecord文件
        writer.write(example.SerializeToString())
        duration_write=time.time()-start
        print('write cost % d'% duration_write)

    writer.close()
    print('Writting End')
    return num_files

def create_multiple_gaussian_map(keypoint_uv, output_size, sigma, valid_vec=None):
    """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
            with variance sigma for multiple coordinates."""
    with tf.name_scope('create_multiple_gaussian_map'):
   # if 1==1:
        #cal palm coor
        palm_coord_uv = tf.expand_dims(0.5*(keypoint_uv[0, :] + keypoint_uv[12, :]), 0)
        keypoint_uv = tf.concat([palm_coord_uv, keypoint_uv[1:21, :]], 0)
        
        #
        #coord_uv_ noise
        noise = tf.truncated_normal([21, 2], mean=0.0, stddev=2.5)
        keypoint_uv += noise
        coords_uv = tf.stack([keypoint_uv[:, 1], keypoint_uv[:, 0]], -1)
        sigma = tf.cast(sigma, tf.float32)
        assert len(output_size) == 2
        s = coords_uv.get_shape().as_list()
        coords_uv = tf.cast(coords_uv, tf.int32)
        if valid_vec is not None:
            valid_vec = tf.cast(valid_vec, tf.float32)
            valid_vec = tf.squeeze(valid_vec)
            cond_val = tf.greater(valid_vec, 0.5)
        else:
            cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
            cond_val = tf.greater(cond_val, 0.5)

        # 筛选数据  在output_size范围内的数据   不能越界
        cond_1_in = tf.logical_and(tf.less(coords_uv[:, 0], output_size[0]-1), tf.greater(coords_uv[:, 0], 0))
        cond_2_in = tf.logical_and(tf.less(coords_uv[:, 1], output_size[1]-1), tf.greater(coords_uv[:, 1], 0))
        cond_in = tf.logical_and(cond_1_in, cond_2_in)
        cond = tf.logical_and(cond_val, cond_in)

        coords_uv = tf.cast(coords_uv, tf.float32)

        # create meshgrid
        x_range = tf.expand_dims(tf.range(output_size[0]), 1)
        y_range = tf.expand_dims(tf.range(output_size[1]), 0)

        X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
        Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

        X.set_shape((output_size[0], output_size[1]))
        Y.set_shape((output_size[0], output_size[1]))

        X = tf.expand_dims(X, -1)
        Y = tf.expand_dims(Y, -1)

        X_b = tf.tile(X, [1, 1, s[0]])
        Y_b = tf.tile(Y, [1, 1, s[0]])

        X_b -= coords_uv[:, 0]
        Y_b -= coords_uv[:, 1]

        dist = tf.square(X_b) + tf.square(Y_b)

        scoremap = tf.exp(-dist / tf.square(sigma)) * tf.cast(cond, tf.float32)

        return scoremap

# 生成整数型的属性
def _float64_feature(value):
    print('float')
    return tf.train.Feature(float64_list=tf.train.FloatList(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
    print('reading images begin')
    start_time = time.time()

    # convert to tfrecords
    readToTfrecord(DATA_DIR)
    # 指定手势图片
    # readToTfrecord(SPECIFY_DATA_DIR)
    duration = time.time() - start_time
    print('convert to tfrecords end , cost %d sec' % duration)


if __name__ == '__main__':
    tf.app.run()
