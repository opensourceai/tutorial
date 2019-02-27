# -*- coding: UTF-8 -*-
# File Name：logistic_regression
# Author : Chen Quan
# Date：2019/2/27
# Description : 逻辑回归
from __future__ import print_function

import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

__author__ = 'Chen Quan'
# 读取数据
X, Y = load_breast_cancer(True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 30])  # 特征个数
y = tf.placeholder(tf.float32, [None])  #

# 设置参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.sigmoid(tf.matmul(x, W) + b)  # 二分类

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 定义初始化参数函数
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 使用session初始化参数
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = X_train[i * batch_size:(i + 1) * batch_size], y_train[
                                                                               i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算精度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
