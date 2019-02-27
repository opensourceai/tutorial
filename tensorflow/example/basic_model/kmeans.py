# -*- coding: UTF-8 -*-
# File Name：kmeans
# Author : Chen Quan
# Date：2019/2/27
# Description : K-means算法
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import os

# 忽略所有GPU，TensorFlow GBDT 不支持 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
__author__ = 'Chen Quan'

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
images = mnist.train.images

# 参数
num_steps = 50  # 训练次数
batch_size = 1024  # 批量大小
k = 25  # 聚类个数
num_classes = 10  # 数字分类个数
num_features = 784  # 图片像素个数即特征个数

# Input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels
Y = tf.placeholder(tf.float32, shape=[None, num_classes])
# K-Means
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

# 创建 KMeans 静态图
training_graph = kmeans.training_graph()

if len(training_graph) > 6:  # Tensorflow 1.4+
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)
# 初始化全局参数
init = tf.global_variables_initializer()

# 创建一个session
sess = tf.Session()

# 使用session正式初始化
sess.run(init, feed_dict={X: images})
sess.run(init_op, feed_dict={X: images})

# Training
for i in range(1, num_steps + 1):

    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: images})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

counts = np.zeros(shape=(k, num_classes))

for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]

labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# 测评
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# 计算精度
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 使用session运行测试
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
