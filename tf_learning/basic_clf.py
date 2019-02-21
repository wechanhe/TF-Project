#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from util import *


def log_reg():
    # 训练LR模型，并进行预测
    model = LogisticRegression(penalty='l2')
    model.fit(train_images, train_labels)
    test_pred = model.predict(test_images)
    print classification_report(test_labels, test_pred), '(logistic regression)'

def gbdt():
    # n_estimators = 90 , min_samples_split=800, max_depth=17, min_samples_leaf=60, subsample=0.9
    # model = GradientBoostingClassifier(random_state=10, n_estimators=90,
    #                                    min_samples_split=800, max_depth=17,
    #                                    min_samples_leaf=60, subsample=0.9)
    model = GradientBoostingClassifier()
    model.fit(train_images, train_labels)
    test_pred = model.predict(test_images)
    print classification_report(test_label, test_pred), '(gdbt)'

def preprocessing():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])

def create_mlp():
    # 构建模型
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(784, )),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # 编译模型
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    model = create_mlp()
    print model.summary()
    model.fit(train_images, train_labels, epochs=30)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    model.save('./model/mlp.h5')

def test():
    # test
    new_model = keras.models.load_model('./model/mlp.h5')
    new_model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    print new_model.summary()
    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# 模型预测
def visualization(model):
    predictions = model.predict(test_images)
    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions,  test_labels)
    plt.show()
    pass

if __name__ == '__main__':
    load_data()
    # train()
    # test()
    # visualization
    new_model = keras.models.load_model('./model/mlp.h5')
    new_model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    print new_model.summary()
    visualization(new_model)