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


# 建立编码网络
def encoder(x):
    layer_1 = tf.keras.layers.Dense(num_hidden_1, activation='sigmoid')(x)
    layer_2 = tf.keras.layers.Dense(num_hidden_2, activation='sigmoid')(layer_1)

    return layer_2


# 建立解码网络
def decoder(x):
    layer_1 = tf.keras.layers.Dense(num_hidden_1, activation='sigmoid')(x)
    layer_2 = tf.keras.layers.Dense(num_input, activation='sigmoid')(layer_1)
    return layer_2


# 创建自解码网络
_input = tf.keras.layers.Input(shape=(num_input,), name='input')
encoder_op = encoder(_input)
decoder_op = decoder(encoder_op)

# 预测
y_pred = decoder_op
# 建立模型
model = tf.keras.Model(_input, y_pred)
# 编译模型：优化器、损失函数、测评函数
model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse, metrics=tf.keras.metrics.mean_squared_error)
# 开始训练
x, y = mnist.train.next_batch(5000)
model.fit(x, x, batch_size=batch_size, epochs=num_steps)

n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
batch_x, batch_y = mnist.validation.next_batch(n)
test_pred = model.predict(batch_x)
for i in range(n):
    # 展示原始图像
    for j in range(n):
        # Draw the original digits
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
    # 展示重构图像
    for j in range(n):
        # Draw the reconstructed digits
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = test_pred[j].reshape([28, 28])

print("原始图像")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("重构图像")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()
