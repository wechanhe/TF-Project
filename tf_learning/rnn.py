#!/usr/bin/python
# -*- coding: UTF-8 -*-

from util import *
import tensorflow as tf

# 导入数据
train_images, train_labels, test_images, test_labels = load_data()

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 10000     # train step 上限
batch_size = 128
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# x y placeholder
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
with tf.name_scope('weight'):
    weights = {
        # shape (28, 128)
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
        # shape (128, 10)
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
    }
with tf.name_scope('biases'):
    biases = {
        # shape (128, )
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
        # shape (10, )
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
    }

def RNN(X, weights, biases):
    '''
    三层：input, cell, output
    :param X:
    :param weights:
    :param biases:
    :return:
    '''
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 使用 basic LSTM Cell.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state,
                                             time_major=False)
    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# 替换成下面的写法:
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    for i in range(training_iters):
        begin = np.random.randint(0, len(train_images) - batch_size)
        batch_xs = train_images[begin:begin+batch_size]
        batch_ys = train_labels[begin:begin+batch_size]
        print batch_xs.shape
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        batch_ys = batch_ys.reshape([-1, 10])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
