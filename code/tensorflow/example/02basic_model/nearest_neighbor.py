# -*- coding: UTF-8 -*-
# File Name：nearest_neighbor
# Author : Chen Quan
# Date：2019/2/27
# Description : 最近邻算法
from __future__ import print_function

__author__ = 'Chen Quan'

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 读取数据
mnist = input_data.read_data_sets("/data/", one_hot=True)

# 提取部分
Xtr, Ytr = mnist.train.next_batch(5000)  # 5000个样本，投票者
Xte, Yte = mnist.test.next_batch(200)  # 200 for testing

# 定义占位符
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# 使用L1距离计算距离
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# 预测: 获取距离最小的index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.
# 定义初始化参数函数
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    # 使用session初始化参数
    sess.run(init)

    # 循环整个测试集
    for i in range(len(Xte)):
        # 获取最近邻index
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})

        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), "True Class:", np.argmax(Yte[i]))
        # 计算精度
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print("Accuracy:", accuracy)
