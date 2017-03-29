from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np

"""
    Defines the Network.
    Contains various methods to add different layers, intended to be as configurable and general as possible. 

     Properties :
            - input : the input tensor
            - batch_size : a scalar, the first dimension of the input tensor which is the size of a batch used in the learning phase
            - layers : a list of tensors representing the successive layers of neurons of the net. layers[0] is the input.
            - variables : a list of tf.Variables which are to be tuned in the learning phase.
            - number_of_layers : self-explanatory...

    Methods :
            - __init__ : initialization
            - add_FCLayer : add a full connected neural layer at the end of the net
            - add_conv_Layer : add a convolution layer at the end of the net
            - compute_output : compute the output of the network by applying the softmax function to its last layer.
"""
class Network:
     
    """
        Initialization of the network.
        Args :
            - input : a tensor of the inputs of the network
       
    """
    def __init__(self,input):
        self.input = input
        self.batch_size = input._shape[0].value
        self.layers = [self.input]
        self.variables = []
        self.number_of_layers = 0

    def add_FCLayer(self,layer_size):
        in_size = self.layers[-1]._shape.as_list()
        layer_size = [self.batch_size]+layer_size
        with tf.name_scope('FC_'+str(self.number_of_layers)):
            W = tf.Variable(initial_value=tf.random_normal(shape=in_size[1:]+layer_size),
                            dtype=tf.float32,
                            name='Weights_FC_'+str(self.number_of_layers))
            b = tf.Variable(initial_value=np.zeros(shape=layer_size),
                            dtype=tf.float32,
                            name='Bias_FC_'+str(self.number_of_layers))
            h = tf.nn.relu(features=tf.tensordot(self.layers[-1],
                                                 W,
                                                 axes=len(in_size)-1)
                           +b,
                           name='FC_'+str(self.number_of_layers))
            h.set_shape(layer_size)
            self.variables.append(W)
            self.variables.append(b)
            self.layers.append(h)
        self.number_of_layers += 1

    def add_conv_Layer(self,kernel_size,stride,padding,out_depth):
        with tf.name_scope('conv_'+str(self.number_of_layers)):
            in_depth = self.layers[-1]._shape[-1].value
            F = tf.Variable(initial_value=tf.random_normal(shape=kernel_size+[in_depth]+[out_depth]),
                            dtype=tf.float32,
                            name = 'Filter_conv_'+str(self.number_of_layers))
            b = tf.Variable(initial_value=tf.zeros(shape=self.layers[-1]._shape.as_list()[:-1]+[out_depth]),
                            dtype = tf.float32,
                            name = 'Bias_conv_'+str(self.number_of_layers))
            h = tf.nn.relu( tf.nn.conv2d(input = self.layers[-1],
                                         filter=F,
                                         strides=stride,
                                         padding=padding)+b,
                           name = 'Conv_'+str(self.number_of_layers))
            self.variables.append(F)
            self.variables.append(b)
            self.layers.append(h)
        self.number_of_layers += 1

    def add_MaxPool_Layer(self,factor=2):
        with tf.name_scope('Max_Pool_'+str(self.number_of_layers)):
            h = tf.nn.max_pool(self.layers[-1],
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1],
                               name='Max_Pool_Layer_'+str(self.number_of_layers),
                               padding='SAME')
            self.layers.append(h)
        self.number_of_layers += 1

    def add_deconv_Layer(self,factor=2):
        with tf.name_scope('Deconv_Layer_'+str(self.number_of_layers)):


    def compute_output(self):
        with tf.name_scope('Output'):
            self.output = tf.nn.softmax(logits=self.layers[-1])

