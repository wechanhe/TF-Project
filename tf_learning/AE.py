#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class AutoEncoder:
    def __init__(self, n_inputs, n_hidden, lr, epochs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_inputs
        self.learning_rate = lr
        self.epochs = epochs

    def train(self, train, test):
        # 模型定义
        input = tf.placeholder(dtype=tf.float16, shape=(None, self.n_inputs))
        hidden = fully_connected(inputs=input, num_outputs=self.n_hidden, activation_fn=None)
        output = fully_connected(inputs=hidden, num_outputs=1, activation_fn=None)
        loss = tf.reduce_mean(tf.square(output - input))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer.minimize(loss)

        # 训练
        encoder = hidden
        loss_ = []
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(self.epochs):
                sess.run(loss, feed_dict={input: train})
                print input, output, loss
                loss_.append(loss)
            encode_val = encoder.eval(feed_dict={input: test})
        return encode_val, loss_

def visualization(features, dim = 3):
    print features[0]
    fig = plt.figure()
    if dim == 3:
        ax = Axes3D(fig=fig)
        ax.scatter3D(features[:, 0], features[:, 1], features[:, 2], color='r')
    elif dim == 2:
        plt.scatter(features[:, 0], features[:, 1])
    plt.show()

def visualize_loss(loss, epochs):
    print loss[0:10], epochs[0:10]
    plt.plot(epochs, loss)
    plt.show()

if __name__ == '__main__':
    ae = AutoEncoder(n_inputs=3, n_hidden=2, lr=0.01, epochs=100)
    train = np.random.randint(low=0, high=10, size=[1000, 3])
    test = np.random.randint(low=0, high=10, size=[200, 3])
    encode_val, loss = ae.train(train, test)
    visualization(test)
    visualization(encode_val, 2)
    # encode_val, loss = ae.train(train, test)
    # visualize_loss(list(loss), [i+1 for i in range(100)])