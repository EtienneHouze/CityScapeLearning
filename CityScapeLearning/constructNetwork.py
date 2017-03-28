from __future__ import print_function

import tensorflow as tf

"""
    Defines the network 
"""
class network:
     
    def __init__(self,inputsize=[1920,1080]):
        self.input_image = tf.placeholder(dtype=tf.float32,shape=(None,tuple(inputsize))
