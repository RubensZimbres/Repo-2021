from itertools import product
import torch as th
m=3
n=3

x = [[list(i[x:x+m]) for x in range(0, len(i), m)] for i in product("01", repeat=m*n)]

kernel=th.tensor(np.matrix(x[199]).astype(int))

central=th.remainder(th.sum(kernel),19)
kernel[1][1]=central

import numpy as np

for i in range(0,len(x)):
    print(np.mod(np.matrix(x[i]).astype(int).sum(),2))

import tensorflow as tf

kernel=tf.convert_to_tensor(np.matrix(x[199]).astype(int))


central=tf.truncatemod(tf.reduce_sum(kernel),2)
kernel.numpy()[1][1]=central
