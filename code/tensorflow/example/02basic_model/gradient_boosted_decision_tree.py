# -*- coding: UTF-8 -*-
# File Name：gradient_boosted_decision_tree
# Author : Chen Quan
# Date：2019/2/27
# Description : 决策树集成算法
from __future__ import print_function, division

import os

import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner
from tensorflow.examples.tutorials.mnist import input_data

__author__ = 'Chen Quan'
# 忽略所有GPU，TensorFlow GBDT 不支持 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.logging.set_verbosity(tf.logging.ERROR)

# 加载数据集
mnist = input_data.read_data_sets("data/", one_hot=False)

# 设置参数
batch_size = 4096  # 每次训练时，批量大小
num_classes = 10  # 数字分类个数
num_features = 784  # 图像像素个数
max_steps = 10000  # 最大迭代次数
# GBDT 参数
learning_rate = 0.1  # 学习率，步长
l1_regul = 0.  # L1正则化系数
l2_regul = 1.  # L2正则化系数
examples_per_layer = 1000
num_trees = 10
max_depth = 16

# 设置config proto
learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_regul / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth
growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER

learner_config.growing_mode = growing_mode
run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
learner_config.multi_class_strategy = (
    gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)

# 创建TensorFlowGBDT集成算法
gbdt_model = GradientBoostedDecisionTreeClassifier(
    model_dir=None,  # No save directory specified
    learner_config=learner_config,
    n_classes=num_classes,
    examples_per_layer=examples_per_layer,
    num_trees=num_trees,
    center_bias=False,
    config=run_config)

# 设置显示日志
tf.logging.set_verbosity(tf.logging.INFO)

# 定义训练的输入函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# 训练模型
gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)

# 评测模型
# 定义评测的输入函数
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# 使用evaluate进行测评
e = gbdt_model.evaluate(input_fn=input_fn)

print("Testing Accuracy:", e['accuracy'])
