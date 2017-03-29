"""
    This is the main script of the project.    
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import join
from os import listdir
from Network import *
from helpers import *

batch_size = 10

def loss(net):
    return 0


with tf.Graph().as_default():
    test_input = tf.placeholder(shape=(batch_size,1920,1080,3),dtype=tf.float32)
    test = Network(test_input)
    test.add_FCLayer(layer_size = [512,512,16])
    #test.add_FCLayer(layer_size = [1024,16])
    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=32)
    test.add_MaxPool_Layer()
    a=0

