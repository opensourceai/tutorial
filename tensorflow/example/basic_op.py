# -*- coding: UTF-8 -*-
# File Name：basic_op
# Author : Chen Quan
# Date：2019/2/26
# Description : 基础的运算操作
from __future__ import print_function
import tensorflow as tf
import numpy as np

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

a_numpy = np.random.randn(4, 3)
b_numpy = np.random.randn(3, 5)
a = tf.constant(a_numpy)
b = tf.constant(b_numpy)
"""
矩阵运算
"""
with tf.Session() as sess:
    print("a=", a_numpy, "b=", b_numpy)
    print("矩阵乘法操作: %i" % sess.run(a @ b))  # Since python >= 3.5 the @ operator is supported (see PEP 465).
    print("矩阵乘法操作: %i" % sess.run(tf.matmul(a, b)))

"""
使用占位符传输数据
"""
#
a = tf.placeholder(tf.int8, shape=[1])
b = tf.placeholder(tf.int8, shape=[1])

with tf.Session() as sess: 
    print("a=", a_numpy, "b=", b_numpy)
    print("矩阵乘法操作: %i" % sess.run(a @ b))  # Since python >= 3.5 the @ operator is supported (see PEP 465).
    print("矩阵乘法操作: %i" % sess.run(tf.matmul(a, b)))
