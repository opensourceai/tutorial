# -*- coding: UTF-8 -*-
# File Name：eager_api
# Author : Chen Quan
# Date：2019/2/27
# Description : enable_eager_execution ，动态图模式
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

__author__ = 'Chen Quan'

print("开启动态图模式")
tf.enable_eager_execution()

a = tf.constant(4)
b = tf.constant(2)
print("a=", a)
# 未使用tf.Session()
print("a * b={}".format((a * b).numpy()))  # a * b=8

# 与numpy结合
c = np.array(2)
print("a * c={}".format(a * c))

# 直接与python基本数据类型结合
d = [4, 3]
print('a * b={}'.format(a * d))
