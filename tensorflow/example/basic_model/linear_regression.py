# -*- coding: UTF-8 -*-
# File Name：linear_regression
# Author : Chen Quan
# Date：2019/2/27
# Description : 线性回归 Y = WX + b
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Chen Quan'

# Parameters
param = {
    "learning_rate": 0.01,
    "training_epochs": 1000,
    "display_step": 50
}
# 训练数据集
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 设置参数
W = tf.Variable(initial_value=tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1, ]), name="bias")

# 模型
pred = tf.add(tf.multiply(X, W), b)  # 与pred=X * W + b等价

# 损失函数：均方误差
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

# 定义优化器
opt = tf.train.GradientDescentOptimizer(param['learning_rate']).minimize(cost)

# 定义初始化参数函数
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 使用session初始化参数
    sess.run(init)

    # 喂入数据
    for epoch in range(param['training_epochs']):
        for (x, y) in zip(train_X, train_Y):
            sess.run(opt, feed_dict={X: x, Y: y})

        if (epoch + 1) % param['display_step'] == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    # 训练误差
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    # 测试集数据
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
