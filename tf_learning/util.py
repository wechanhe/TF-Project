#!/usr/bin/python 
# -*- coding: UTF-8 -*-
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = []
train_labels = []
test_images = []
test_labels = []

def load_data():
    global train_images, train_labels, test_images, test_labels
    data = input_data.read_data_sets('data/fashion')
    train_images = data.train.images / 255.0
    train_labels = data.train.labels
    test_images = data.test.images / 255.0
    test_labels = data.test.labels
    print 'train images:', len(train_images), 'train labels:', len(train_labels)
    print 'test images:', len(test_images), 'test labels', len(test_labels)
    return train_images, train_labels, test_images, test_labels


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.reshape(28,28), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def save_model(model, model_name):
    print 'save model', model_name
    s = pickle.dumps(model)
    f = open('./model/' + model_name, 'w')
    f.write(s)
    f.close()

def read_model(model_name):
    print 'read model', model_name
    f2 = open('./model/' + model_name, 'r')
    s2 = f2.read()
    model = pickle.loads(s2)
    return model
