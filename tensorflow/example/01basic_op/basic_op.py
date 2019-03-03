# -*- coding: UTF-8 -*-
# File Name：01basic_op
# Author : Chen Quan
# Date：2019/2/26
# Description : 基础的运算操作
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.set_random_seed(2019)
__author__ = 'Chen Quan'

"""
标量运算
"""

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a=2, b=3")
    print("加法操作: %i" % sess.run(a + b))
    print("加法操作: %i" % sess.run(tf.add(a, b)))

    print("减法操作: %i" % sess.run(a - b))
    print("减法操作: %i" % sess.run(tf.subtract(a, b)))

    print("乘法操作: %i" % sess.run(a * b))
    print("乘法操作: %i" % sess.run(tf.multiply(a, b)))

    print("除法操作: %i" % sess.run(a / b))
    print("除法操作: %i" % sess.run(tf.divide(a, b)))

"""
矩阵运算
"""

a_numpy = np.random.randn(4, 3)
b_numpy = np.random.randn(3, 5)
c_numpy = np.random.randn(4, 3)

a = tf.constant(a_numpy)
b = tf.constant(b_numpy)
c = tf.constant(c_numpy)

with tf.Session() as sess:
    print("a", a)
    print("b", b)
    print("c", c)
    print("=================")

    print("矩阵点乘法操作: %i" % sess.run(a @ b))  # Since python >= 3.5 the @ operator is supported (see PEP 465).
    print("矩阵点乘法操作: %i" % sess.run(tf.matmul(a, b)))

    # 两个矩阵对应元素相乘
    print("矩阵叉乘法操作: %i" % sess.run(a * c))
    print("矩阵叉乘法操作: %i" % sess.run(tf.multiply(a, c)))

    # 两个矩阵对应元素相除
    print("矩阵除法操作: %i" % sess.run(a / c))
    print("矩阵除法操作: %i" % sess.run(tf.divide(a, c)))

    # 两个矩阵对应元素相加
    print("矩阵加法操作：%i" % sess.run(a + c))
    print("矩阵加法操作：%i" % sess.run(tf.add(a, c)))

    # 两个矩阵对应元素相减
    print("矩阵加法操作：%i" % sess.run(a - c))
    print("矩阵加法操作：%i" % sess.run(tf.subtract(a, c)))
a_numpy = np.random.randn(4, 2)
b_numpy = np.random.randn(4, 1)
c_numpy = np.random.randn(1, 2)
a = tf.constant(a_numpy)
b = tf.constant(b_numpy)
c = tf.constant(c_numpy)
with tf.Session() as sess:
    print("a", a)
    print("b", b)
    print("c", c)

    # 叉乘广播 a.shape=[4,2]
    print("矩阵叉乘法操作: %i" % sess.run(a * b))  # b.shape=[4,1]
    print("矩阵叉乘法操作: %i" % sess.run(tf.multiply(a, b)))

    print("矩阵叉乘法操作: %i" % sess.run(a * c))  # c.shape=[1,2]
    print("矩阵叉乘法操作: %i" % sess.run(tf.multiply(a, c)))

    # 除法的广播
    print("矩阵除法操作: %i" % sess.run(a / b))
    print("矩阵除法操作: %i" % sess.run(tf.divide(a, b)))

    print("矩阵除法操作: %i" % sess.run(a / c))
    print("矩阵除法操作: %i" % sess.run(tf.divide(a, c)))

    # 矩阵加法的广播
    print("矩阵加法操作：%i" % sess.run(a + b))
    print("矩阵加法操作：%i" % sess.run(tf.add(a, b)))

    print("矩阵加法操作：%i" % sess.run(a + c))
    print("矩阵加法操作：%i" % sess.run(tf.add(a, c)))

    # 减法的广播
    print("矩阵加法操作：%i" % sess.run(a - b))
    print("矩阵加法操作：%i" % sess.run(tf.subtract(a, b)))

    print("矩阵加法操作：%i" % sess.run(a - c))
    print("矩阵加法操作：%i" % sess.run(tf.subtract(a, c)))

"""
使用占位符传输数据
"""
# 创建占位符，形状为[1,]
a = tf.placeholder(tf.int8, shape=[1])  # 传入一个8位整数
b = tf.placeholder(tf.float32, shape=[1])  # 传入一个32位浮点数
c = tf.placeholder(tf.int8, shape=[1])
with tf.Session() as sess:
    print("a=", a, "b=", b)
    b = a * b
    print('a={}'.format(sess.run([a], feed_dict={a: [1]})))
    print('b={}'.format(sess.run([b], feed_dict={b: [1.1]})))
    try:
        print('a * b={}'.format(sess.run([a * b], feed_dict={b: [1.1]})))
    except TypeError:
        print("错误")
    print('a*c={}'.format(sess.run([a * c], feed_dict={a: [1], b: [2]})))
