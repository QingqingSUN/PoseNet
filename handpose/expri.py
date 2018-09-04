import tensorflow as tf

import numpy as np


sess=tf.InteractiveSession()
kwy=np.array(range(42))
ad=np.sum(kwy)/42
print('ad ',ad)
kwy=kwy/ad

kwy[0]=-2
print(kwy)

kwy=tf.convert_to_tensor(kwy,dtype=tf.float32)
kwy=tf.reshape(kwy,(21,2))
print(kwy[0][:])
m=0.5*(kwy[0, :] + kwy[12, :])
print(m.get_shape().as_list())
palm_coord_uv = tf.expand_dims(0.5*(kwy[0, :] + kwy[12, :]), 0)
print(palm_coord_uv.get_shape().as_list())
print(palm_coord_uv.eval())
keypoint_uv = tf.concat([palm_coord_uv, kwy[1:21, :]], 0)
print(keypoint_uv.get_shape().as_list())
print(keypoint_uv.eval())
coords_uv=tf.stack([keypoint_uv[:,1],keypoint_uv[:,0]],1)
s = coords_uv.get_shape().as_list()
print(coords_uv.eval())
print(coords_uv.get_shape().as_list())
u=tf.unstack(keypoint_uv,axis=1)
print(u[0].eval())

cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
print(cond_val.eval())
print(cond_val.get_shape().as_list())


cond_val = tf.greater(cond_val, 0.5)
print(cond_val.eval())
output_size=[4,5]
cond_1_in = tf.logical_and(tf.less(coords_uv[:, 0], output_size[0]-1), tf.greater(coords_uv[:, 0], 0))
print(cond_1_in.eval())
cond_2_in = tf.logical_and(tf.less(coords_uv[:, 1], output_size[1]-1), tf.greater(coords_uv[:, 1], 0))
print(cond_2_in.eval())
cond_in = tf.logical_and(cond_1_in, cond_2_in)
cond = tf.logical_and(cond_val, cond_in)

print(cond.eval())

coords_uv = tf.cast(coords_uv, tf.float32)

x_range = tf.expand_dims(tf.range(output_size[0]), 1)
y_range = tf.expand_dims(tf.range(output_size[1]), 0)

X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)
print(X.eval())
print(Y.eval())
X.set_shape((output_size[0], output_size[1]))
Y.set_shape((output_size[0], output_size[1]))

X = tf.expand_dims(X, -1)
Y = tf.expand_dims(Y, -1)

X_b = tf.tile(X, [1, 1, s[0]])
Y_b = tf.tile(Y, [1, 1, s[0]])
print(X_b.get_shape().as_list())

print(X_b.eval())
print(Y_b.eval())
X_b -= coords_uv[:, 0]
Y_b -= coords_uv[:, 1]
print(X_b.eval())
print(Y_b.eval())
dist = tf.square(X_b) + tf.square(Y_b)

scoremap = tf.exp(-dist / tf.square(25.0)) * tf.cast(cond, tf.float32)
print(scoremap.get_shape().as_list())
print(scoremap.eval())