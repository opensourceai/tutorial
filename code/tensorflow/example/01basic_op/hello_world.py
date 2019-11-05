# -*- coding: UTF-8 -*-
# File Name：hello_world
# Author : Chen Quan
# Date：2019/2/26
# Description : 一个简单的Hello world例子
from __future__ import print_function

import tensorflow as tf

__author__ = 'Chen Quan'

"""
创建一个常量操作
将op作为节点添加到默认图形中
＃运行后返回的值表示常量操作的输出
"""
# 创建常量操作
hello = tf.constant('Hello, TensorFlow!')
# 创建TF会话
sess = tf.Session()

# 执行操作
print(sess.run(hello))
