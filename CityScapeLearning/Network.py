from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np
import helpers


class Network:
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
    def __init__(self,input,name='net'):
        """
        Initialization of the network.
        Args :
            - input : a tensor of the inputs of the network
       
        """
        self.input = input
        self.batch_size = input.get_shape()[0].value
        self.last_layer = self.input
        self.encoding_layers = []
        self.encoder_variables = []
        self.decoder_variables = []
        self.name = name

    def add_FCLayer(self,layer_size,relu=True):
        """
            Deprecated
        """
        in_size = self.layers[-1].get_shape().as_list()
        layer_size = [self.batch_size]+layer_size
        with tf.name_scope(self.name):
            with tf.name_scope('FC_'+str(self.number_of_layers)):
                W = tf.Variable(initial_value=tf.truncated_normal(shape=in_size[1:]+layer_size,stddev=10./layer_size),
                                dtype=tf.float32,
                                name='Weights_FC_'+str(self.number_of_layers))
                b = tf.Variable(initial_value=tf.random_uniform(shape=layer_size),
                                dtype=tf.float32,
                                name='Bias_FC_'+str(self.number_of_layers))
                if(relu):
                    h = tf.nn.relu(features=tf.tensordot(self.layers[-1],
                                                         W,
                                                         axes=len(in_size)-1)
                                   +b,
                                   name='FC_'+str(self.number_of_layers))
                else:
                    h = tf.add(tf.tensordot(self.layers[-1],
                                            W,
                                            axes = len(in_size)-1),
                               b,
                               name='FC_'+str(self.number_of_layers))
                h.set_shape(layer_size)
                self.variables.append(W)
                self.variables.append(b)
                self.layers.append(h)
            self.number_of_layers += 1


    def add_conv_Layer(self,kernel_size,stride,padding,out_depth,relu=True):
        """
            Deprecated
        """
        with tf.name_scope(self.name):
            with tf.name_scope('conv_'+str(self.number_of_layers)):
                in_depth = self.layers[-1].get_shape()[-1].value
                F = tf.Variable(initial_value=tf.truncated_normal(shape=kernel_size+[in_depth]+[out_depth],stddev=10./out_depth),
                                dtype=tf.float32,
                                name = 'Filter_conv_'+str(self.number_of_layers))
                h = tf.nn.conv2d(input = self.layers[-1],
                                                 filter=F,
                                                 strides=stride,
                                                 padding=padding)
                b = tf.Variable(initial_value=tf.zeros(h.get_shape()[1:]),
                                dtype = tf.float32,
                                name = 'Bias_conv_'+str(self.number_of_layers))

                if(relu):
                    h = tf.nn.relu( tf.nn.conv2d(input = self.layers[-1],
                                                 filter=F,
                                                 strides=stride,
                                                 padding=padding)+b,
                                   name = 'Conv_'+str(self.number_of_layers))
                else:
                    h = tf.add(tf.nn.conv2d(input = self.layers[-1],
                                                 filter=F,
                                                 strides=stride,
                                                 padding=padding),
                               b,
                               name = 'Conv_'+str(self.number_of_layers))
                self.variables.append(F)
                self.variables.append(b)
                self.layers.append(h)
            self.number_of_layers += 1

    def add_MaxPool_Layer(self,factor=2):
        """
            Deprecated
        """
        with tf.name_scope(self.name):
            with tf.name_scope('Max_Pool_'+str(self.number_of_layers)):
                self.last_layer = tf.nn.max_pool(self.last_layer,
                                   ksize=[1,2,2,1],
                                   strides=[1,2,2,1],
                                   name='Max_Pool_Layer_'+str(self.number_of_layers),
                                   padding='SAME')

    def add_deconv_Layer(self,out_depth,relu=True):
        """
            Deprecated
        """
        with tf.name_scope(self.name):
            with tf.name_scope('Deconv_Layer_'+str(self.number_of_layers)):
                in_depth = self.layers[-1]._shape[-1].value
                out_height = 2 * self.layers[-1]._shape[1].value
                out_width = 2 * self.layers[-1]._shape[2].value
                F = tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,out_depth,in_depth],
                                                                  stddev=0.1,
                                                                  dtype=tf.float32),
                                name='Filter_Deconv_'+str(self.number_of_layers))
                if(relu):
                    h = tf.nn.relu(tf.nn.conv2d_transpose(value = self.layers[-1],
                                               filter=F,
                                               output_shape = [self.batch_size,out_height, out_width, out_depth],
                                               strides = [1, 2 , 2, 1],
                                               padding = "SAME"),
                                   name = 'Deconv_Layer_'+str(self.number_of_layers))
                else:
                     h = tf.nn.conv2d_transpose(value = self.layers[-1],
                                               filter=F,
                                               output_shape = [self.batch_size,out_height, out_width, out_depth],
                                               strides = [1, 2 , 2, 1],
                                               padding = "SAME",
                                               name = 'Deconv_Layer_'+str(self.number_of_layers))
                self.variables.append(F)
                self.layers.append(h)
            self.number_of_layers += 1

    def compute_output(self,top1=True):
        """
            Computes the output of the network.
            @ args :
                - top1 : indicates whether to keep only the top1 prediction or not.
            @ does :
                - updates the output tensor of the network
        """
        with tf.name_scope('Output'):
            if(top1):
                self.output = tf.argmax(tf.nn.softmax(self.last_layer),
                                        axis = len(self.last_layer.get_shape()[1:].as_list()),
                                        name='output')
            else:
                self.output = tf.nn.softmax(logits=self.last_layer,
                                        name='output')

    
    def add_complete_encoding_layer(self,depth,layerindex,bias=True,num_conv=2,ksize=[3,3],pooling=True,relu=True):
        """
        Adds a complete encoding layer to the network
            @ args :
                - depth : the depth of the convolutional layer
                - layerindex : the index for the layer, keep it clear !
                - bias : a boolean specifying whether to use bias or not for the convolution
                - num_conv = the number of convolutions to perform in the layer
                - ksize = the size of the convolution filter
                - pooling : a  boolean specifying whether to maxpool the output of this layer or not.
                - relu : indicates whether to use the relu function of the output or not.
            @ does :
                - adds bias and filters variables to the list of encoding variables of the net
                - updates self.last_layer with the output of this layer.
                - appends the before pooling layer to the encoding_layers list of the net.
        """
        in_shape = self.last_layer.get_shape()[1:].as_list()
        with tf.name_scope('Variables_Encoding_'+str(layerindex)):
            F = []
            if (bias):
                B = []
            for i in range(num_conv):
                F.append(tf.Variable(initial_value=tf.truncated_normal(shape=[ksize[0],ksize[1],in_shape[-1],depth],
                                                                       stddev=10./depth,
                                                                    dtype = tf.float32,
                                                                    name = 'Filter_'+str(layerindex)+'_'+str(i)
                                                                    )
                                        )
                            )
                if (bias):
                    B.append(tf.Variable(initial_value=tf.random_uniform(minval=-1,maxval=1,shape=in_shape[:-1]+[depth]),
                            dtype = tf.float32,
                            name = 'Bias_'+str(layerindex)+'_'+str(i))
                                )
                in_shape[-1] = depth
        with tf.name_scope('Encoding_'+str(layerindex)):
            for i in range(num_conv):
                in_shape = self.last_layer.get_shape()[1:].as_list()
                if (bias):
                    if(relu):
                        self.last_layer = tf.nn.relu(tf.nn.conv2d(self.last_layer,F[i],strides=[1,1,1,1],padding="SAME")+B[i])
                    else:
                        self.last_layer = tf.nn.conv2d(self.last_layer,F[i],strides=[1,1,1,1],padding="SAME")+B[i]
                else:
                    if(relu):
                        self.last_layer = tf.nn.relu(tf.nn.conv2d(self.last_layer,F[i],strides=[1,1,1,1],padding="SAME"))
                    else : 
                        self.last_layer = tf.nn.conv2d(self.last_layer,F[i],strides=[1,1,1,1],padding="SAME")
                if(bias):
                    self.encoder_variables.extend(B+F)
                else:
                    self.encoder_variables.extend(F)
                self.last_layer = tf.layers.batch_normalization(self.last_layer)
            self.encoding_layers.append(self.last_layer)
            if (pooling):
                self.last_layer = tf.nn.max_pool(self.last_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name = 'pooling')

    def add_complete_decoding_layer(self,corresponding_encoding,bias=False,num_conv=0,ksize=3, init_weight = 0.5):
        """
            Appends a complete decoding layer to the network.
            @ args :
                - corresponding_encoding : the index of the encoding layer whose input will be used to compute the output, according to the CNN architecture.
                - bias : whether to use bias or not
                - num_conv : the number of convolutions to perform after the unpooling/fusion operation
                - ksize : kernel size to use for the convolutions
                - init_weight : the initial weight given to the layer coming from the corresponding encoding layer
            @ does :
                - updates the last_layer of the network
                - adds the relevant variables to the decoder_variables list of the net
        """
        in_shape = self.last_layer.get_shape().as_list()
        depth = in_shape[-1]
        with tf.name_scope('Variables_Decoding_'+str(corresponding_encoding)):
            F = []
            deconv_filter = tf.Variable(initial_value=tf.truncated_normal(shape=[1,1,self.encoding_layers[corresponding_encoding].get_shape()[-1].value,depth],
                                                                          stddev=10./depth),
                                        dtype=tf.float32,
                                        name='Unpooling_Filter')
            deconv_weights = tf.Variable(initial_value= init_weight * tf.ones(shape = [2*in_shape[1],2*in_shape[2],depth]),
                                         dtype = tf.float32,
                                         name = 'Unpooling_Weights')
            helpers.variable_summaries(deconv_weights)
            if (bias):
                B=[]
            for i in range(num_conv):
                F.append(tf.Variable(initial_value=tf.truncated_normal(shape=[ksize,ksize,depth,depth],
                                                                       stddev=1./depth,
                                                                       dtype = tf.float32,
                                                                       name = 'Filter_Deocde'+str(corresponding_encoding)+'_'+str(i)
                                                                       )
                                        )
                            )
                if (bias):
                    B.append(tf.Variable(initial_value=tf.random_uniform([in_shape[0],2*in_shape[1],2*in_shape[2],depth],
                                                                         minval=-1,
                                                                         maxval=1),
                            dtype = tf.float32,
                            name = 'Bias_'+str(corresponding_encoding)+'_'+str(i))
                                )
        with tf.name_scope('Decoding_'+str(corresponding_encoding)):
            self.last_layer = tf.image.resize_bilinear(self.last_layer,size=[ 2*in_shape[1], 2*in_shape[2]])
            self.encoding_layers[corresponding_encoding] = tf.nn.conv2d(self.encoding_layers[corresponding_encoding],deconv_filter,strides=[1,1,1,1],padding="SAME")
            self.last_layer = tf.add(self.last_layer,tf.multiply(self.encoding_layers[corresponding_encoding],deconv_weights))
            self.decoder_variables.append(deconv_weights)
            for i in range(num_conv):
                if (bias):
                    self.last_layer = tf.nn.relu(tf.nn.conv2d(self.last_layer,F[i],strides=[1,1,1,1],padding="SAME")+B[i])
                else:
                    self.last_layer = tf.nn.relu(tf.nn.conv2d(self.last_layer,F[i],strides=[1,1,1,1],padding="SAME"))
                if(bias):
                    self.decoder_variables.extend(B+F)
                else:
                    self.decoder_variables.extend(F)
                self.last_layer = tf.layers.batch_normalization(self.last_layer)