#
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
import numpy as np
import scipy.misc
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d,load_weights_from_snapshot
from os import walk
from os.path import join


def cac_dis_cos(x, y):
    # num =float(x * y)
    # print(x,y)
    # num=np.multiply(x, y)
    num = np.dot(x, y)
    # print('num=',num)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if(denom==0):
        cos=-1
    else:
        cos = num / denom
    sim = 0.5 + 0.5 * cos
    return cos


def cac_point_cos(x, bons):
    dis = 0.0
    # print(x)
    a0 = x[bons[0][1], :] - x[bons[0][0], :]
    # print('a0',a0)
    b0 = x[bons[1][1], :] - x[bons[1][0], :]
    # print('b0',b0)
    zero_cos=cac_dis_cos(a0,b0)
    # dis+=zero_cos
    print('zero_cos',zero_cos)

    a1 = x[bons[1][1], :] - x[bons[1][0], :]
    # print('a1', a1)
    b1 = x[bons[2][1], :] - x[bons[2][0], :]
    # print('b1', b1)
    one_cos = cac_dis_cos(a1, b1)
    # dis += one_cos
    print('one_cos', one_cos)

    a2 = x[bons[2][1], :] - x[bons[2][0], :]
    # print('a2', a2)
    b2 = x[bons[3][1], :] - x[bons[3][0], :]
    # print('b2', b2)
    two_cos = cac_dis_cos(a2, b2)
    # dis += two_cos
    print('two_cos', two_cos)

#制定规则
    if(one_cos>0.9):
        two_cos=1
    if(one_cos<0.5):
        two_cos=0
    dis=zero_cos+one_cos+two_cos
    print('dis', dis)
    return dis


def cac_kp_feat(x, bons):
    dis = []
    bon = list(bons)
    # print(bon)
    for i in range(0, 5):
        if(i==0):
            print('thumb')
        elif(i==1):
            print('second')
        elif(i==2):
            print('middle')
        elif(i==3):
            print('ring')
        elif(i==4):
            print('little')
        i = i * 4
        bons_np = bon[i:i + 4]
        # print('bons_np=',bons_np)
        dis_one = cac_point_cos(x, bons_np)
        dis.append(dis_one)
    return dis


PATH_TO_SAVE='E:/python/ServerCode/Data/cropplot/'
PATH_TO_IMAGE = 'E:/python/ServerCode/Data/croptest/'
PATH_TO_SNAPSHOTS = 'snapshots_posenet'

if __name__ == '__main__':
    bones = [(0, 4),
             (4, 3),
             (3, 2),
             (2, 1),

             (0, 8),
             (8, 7),
             (7, 6),
             (6, 5),

             (0, 12),
             (12, 11),
             (11, 10),
             (10, 9),

             (0, 16),
             (16, 15),
             (15, 14),
             (14, 13),

             (0, 20),
             (20, 19),
             (19, 18),
             (18, 17)]


    # PATH_TO_SNAPSHOTS = '/snapshots_posenet/'
    image_list = list()
    # image_list.append('./data/q1.jpg')
    # image_list.append('./data/test2/q2.png')
    alllist = os.listdir(PATH_TO_IMAGE)


    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))

    # build network
    net = ColorHandPose3DNetwork()

    keypoints_scoremap_tf = net.inference_pose2d(image_tf)
    keypoints_scoremap_tf = keypoints_scoremap_tf[-1]
    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    last_cpt = tf.train.latest_checkpoint(PATH_TO_SNAPSHOTS)
    last_cpt = os.path.join(PATH_TO_SNAPSHOTS, 'model-13')
    print('last_cpt', last_cpt)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

    for img_name in alllist:
        print('img_name',img_name)
        # os.path.join(PATH_TO_IMAGE,img_name)
        image_raw = scipy.misc.imread(os.path.join(PATH_TO_IMAGE,img_name))



        image_crop_v = scipy.misc.imresize(image_raw, (256, 256))
        image_v = np.expand_dims((image_crop_v.astype('float') / 255.0) - 0.5, 0)

        keypoints_scoremap_v = sess.run(keypoints_scoremap_tf, feed_dict={image_tf: image_v})

        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        image_crop_v = np.squeeze(image_crop_v)

        # post processing
        image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v)) * 8

        x = coord_hw_crop
        # print('keypoints',x)
        dis = cac_kp_feat(x, bones)
        # out.write(img.split('/')[-1])

        print(dis)

        #给定各手指过滤阈值
        yesorno=[dis[0]>1.52,dis[1]>2.2,dis[2]>2.57,dis[3]<1.32,dis[4]<1.83]
        print(yesorno)
        # yes=(int)(yesorno)
        yes=sum(yesorno)
        if(yes==5):
            print('yes')
        else:
            print('no')

       #绘制关键点图片并保存
        tempimage = image_raw

        plt.imshow(tempimage)
        plot_hand(coord_hw_crop, plt)

        plt.savefig(os.path.join(PATH_TO_SAVE,img_name))
        plt.close()