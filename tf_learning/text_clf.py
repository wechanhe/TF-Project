#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from util import *

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
