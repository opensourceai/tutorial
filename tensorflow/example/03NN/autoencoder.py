# -*- coding: UTF-8 -*-
# File Name：autoencoder
# Author : Chen Quan
# Date：2019/2/27
# Description : 自解码

__author__ = 'Chen Quan'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取数据
mnist = input_data.read_data_sets("data/", one_hot=True)

# 设置训练参数
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# 自解码网络参数
num_hidden_1 = 256  # 第一层神经单元个数
num_hidden_2 = 128  # 第一层神经单元个数
num_input = 784  #

# 定义占位符
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}


# 建立编码网络
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# 建立解码网络
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# 创建自解码网络
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# 预测
y_pred = decoder_op
# 自解码网络中，输入即为实际输出（Label）
y_true = X

# 定义损失函数：均方误差
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# 定义优化器：RMSProp
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# 定义初始化参数函数
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    # 初始化
    sess.run(init)

    # 训练
    for i in range(1, num_steps + 1):
        # 准备数据
        batch_x, _ = mnist.train.next_batch(batch_size)
        # 给自解码网络喂入数据
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # 测试
    # 对测试集中的图像进行编码和解码，并可视化其重建。
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST测试集
        batch_x, _ = mnist.test.next_batch(n)
        # 对数字图像进行编码和解码
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # 展示原始图像
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
        # 展示重构图像
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    print("原始图像")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("重构图像")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
