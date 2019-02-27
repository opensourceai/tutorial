# -*- coding: UTF-8 -*-
# File Name：random_forest
# Author : Chen Quan
# Date：2019/2/27
# Description : 随机森林
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import resources

__author__ = 'Chen Quan'

import os

# 随机森林不支持GPU，忽略所有GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Import MNIST data

mnist = input_data.read_data_sets("data/", one_hot=False)

# Parameters
num_steps = 500  # 训练次数
batch_size = 1024  # 批量大小
num_classes = 10  # 分类数（数字种类）
num_features = 784  # 像素大小28*28
num_trees = 10
max_nodes = 1000

# 定义占位符
X = tf.placeholder(tf.float32, shape=[None, num_features])
# 对应于随机森林来说，label必须是整数型，即class id为整数
Y = tf.placeholder(tf.int32, shape=[None])
# 初始化随机森林参数
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()
# 构建随机森林
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# 获取静态图和训练日志
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# 计算精度
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化
init = tf.group(tf.global_variables_initializer(),
                resources.initialize_resources(resources.shared_resources()))
# 开始训练
with tf.Session() as sess:
    # 使用session初始化
    sess.run(init)

    # Training
    for i in range(1, num_steps + 1):
        # 准备数据
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 给随机森林喂入数据
        _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        if i % 50 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

    # 测试模型
    test_x, test_y = mnist.test.images, mnist.test.labels
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
