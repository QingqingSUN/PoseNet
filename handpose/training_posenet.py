#!/usr/bin/python
#_*_coding:utf-8_*_
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys
import pose_input
import pickle
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from data.BinaryDbReader import BinaryDbReader
from utils.general import LearningRateScheduler, load_weights_from_snapshot
import numpy as np
# training parameters
train_para = {'lr': [1e-4, 1e-5, 1e-6],
              'lr_iter': [10, 20],
              'max_iter': 30,
              'show_loss_freq': 1,
              'snapshot_freq': 10,
              'snapshot_dir': 'snapshots_posenet'}

# get dataset
#dataset = BinaryDbReader(mode='training',
 #                        batch_size=8, shuffle=True, use_wrist_coord=False,
  #                       hand_crop=True, coord_uv_noise=True, crop_center_noise=True)

#从tfrecords获取数据

images,labels,imagename=pose_input.inputs('train',8,1)

print('image,labels',labels,images)


# build network graph
#data = dataset.get()

# build network
evaluation = tf.placeholder_with_default(True, shape=())


net = ColorHandPose3DNetwork()
#keypoints_scoremap = net.inference_pose2d(data['image_crop'], train=True)


keypoints_scoremap = net.inference_pose2d(images, train=True)


#s = data['scoremap'].get_shape().as_list()
s = labels.get_shape().as_list()

keypoints_scoremap = [tf.image.resize_images(x, (s[1], s[2])) for x in keypoints_scoremap]

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
s = labels.get_shape().as_list()
#s = data['scoremap'].get_shape().as_list()

#vis = tf.cast(tf.reshape(data['keypoint_vis21'], [s[0], s[3]]), tf.float32)
print('keypoints_scoremap',keypoints_scoremap)
for i, pred_item in enumerate(keypoints_scoremap):
    #loss += tf.reduce_sum(vis * tf.sqrt(tf.reduce_mean(tf.square(pred_item - data['scoremap']), [1, 2]))) / (tf.reduce_sum(vis) + 0.001)

    loss += tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(pred_item - labels), [1, 2]))) / (21 + 0.001)

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=4.0)

with open('./weights/posenet3d-rhd-stb-slr-finetuned.pickle', 'rb') as fi:
    weight_dict = pickle.load(fi)
    #print(weight_dict.keys())
    pose2d_dict = {}
    for k in weight_dict:
        if k.find(u'PoseNet2D') > -1:
     #       print(k)
            pose2d_dict[k] = weight_dict[k]
    init_op, init_feed = tf.contrib.framework.assign_from_values(pose2d_dict)
    sess.run(init_op, init_feed)


# rename_dict = {'CPM/PoseNet': 'PoseNet2D',
#                '_CPM': ''}
# load_weights_from_snapshot(sess, './weights/cpm-model-mpii', ['PersonNet', 'PoseNet/Mconv', 'conv5_2_CPM'], rename_dict)
# PATH_TO_SNAPSHOTS ='train1106_snapshots_posenet'
# last_cpt = os.path.join(PATH_TO_SNAPSHOTS, 'model-6000')
# print('last_cpt', last_cpt)
# assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
# load_weights_from_snapshot(sess, last_cpt)
# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):
    _, loss_v,image_name_v = sess.run([train_op, loss,imagename])

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, loss_v))
        print('imagename',image_name_v)
        sys.stdout.flush()
    if(loss_v<1):
       np.savetxt('imagename.csv',image_name_v )
    
    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
