# -*- coding: UTF-8 -*-
# File Name：neural_network
# Author : Chen Quan
# Date：2019/2/28
# Description : 神经网络TensorFlow keras代码实现
__author__ = 'Chen Quan'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
# 设置参数
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# 网络参数
n_hidden_1 = 128  # 第一层神经单元个数
n_hidden_2 = 256  # 第二层神经单元个数
num_input = 784  # 输入层
num_classes = 10  # 输出层神经单元个数


# 建立模型
def neural_net(x):
    x = tf.keras.layers.Dense(n_hidden_1, activation='relu')(x)
    x = tf.keras.layers.Dense(n_hidden_2, activation='relu')(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return x


# 创建模型
_input = tf.keras.Input(shape=(num_input,))
logits = neural_net(_input)

model = tf.keras.Model(_input, logits)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy,
              metrics=tf.keras.metrics.categorical_accuracy)
x, y = mnist.train.next_batch(5000)
# 运行模型
model.fit(x, y, batch_size=batch_size, epochs=num_steps)
test_x, test_y = mnist.validation.next_batch(5000)

evaluate = model.evaluate(test_x, test_y)
print("evaluate", evaluate)
