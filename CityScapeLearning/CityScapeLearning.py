from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import join
from os import listdir
from Network import *
from helpers import *

batch_size = 10

with tf.Graph().as_default():
    test_input = tf.placeholder(shape=(batch_size,10),dtype=tf.float32)
    test = Network(test_input)
    test.add_FCLayer(layer_size = [1024])
    test.add_FCLayer(layer_size = [1024,16])
    a=0

