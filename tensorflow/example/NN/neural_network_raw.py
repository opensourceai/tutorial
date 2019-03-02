# -*- coding: UTF-8 -*-
# File Name：neural_network_raw
# Author : Chen Quan
# Date：2019/2/28
# Description : 神经网络TensorFlow原生代码实现
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

# 占位符，数据入口
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# 初始化参数
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# 建立模型
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# 创建模型
logits = neural_net(X)

prediction = tf.nn.softmax(logits)

# 定义损失函数和优化器
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# 评测模型
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    # 运行初始化
    sess.run(init)

    for step in range(1, num_steps + 1):
        # 准备数据
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # 计算测试集精度
    print("Testing Accuracy:", sess.run(accuracy,
                                        feed_dict={X: mnist.test.images,
                                                   Y: mnist.test.labels}))
