from __future__ import print_function, division, absolute_import

import tensorflow as tf

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

    def __init__(self, input, name='net',numlab = 35):
        """
        Initialization of the network.
        @ Args :
            - input : a 4D tensor of the inputs of the network
            - name : a string
            - numlab : the number of labels of the network
        @ Fields :
            - input : the 4D input tensor
            - output : the batchsize*imH*imW*1 tensor containing the labelled prediction of the input.
            - last_layer : a 4D batchsize*imH*imW*numlab tensor of logits
            - variables : a dict containing lists of tensors storing the different variables to tune. This construction allows fine tuning in the training method.
            
        """
        self.input = input
        self.batch_size = input.get_shape()[0].value
        self.last_layer = self.input
        self.encoding_layers = []
        self.encoder_variables = []
        self.decoder_variables = []
        self.variables = {}
        self.name = name
        self.numlabs = numlab

        """
            DEPRECATED
        def add_FCLayer(self, layer_size, relu=True):
     
            in_size = self.layers[-1].get_shape().as_list()
            layer_size = [self.batch_size] + layer_size
            with tf.name_scope(self.name):
                with tf.name_scope('FC_' + str(self.number_of_layers)):
                    W = tf.Variable(
                        initial_value=tf.truncated_normal(shape=in_size[1:] + layer_size, stddev=10. / layer_size),
                        dtype=tf.float32,
                        name='Weights_FC_' + str(self.number_of_layers))
                    b = tf.Variable(initial_value=tf.random_uniform(shape=layer_size),
                                    dtype=tf.float32,
                                    name='Bias_FC_' + str(self.number_of_layers))
                    if (relu):
                        h = tf.nn.relu(features=tf.tensordot(self.layers[-1],
                                                             W,
                                                             axes=len(in_size) - 1)
                                                + b,
                                       name='FC_' + str(self.number_of_layers))
                    else:
                        h = tf.add(tf.tensordot(self.layers[-1],
                                                W,
                                                axes=len(in_size) - 1),
                                   b,
                                   name='FC_' + str(self.number_of_layers))
                    h.set_shape(layer_size)
                    self.variables.append(W)
                    self.variables.append(b)
                    self.layers.append(h)
                self.number_of_layers += 1

        def add_conv_Layer(self, kernel_size, stride, padding, out_depth, relu=True):
         
            with tf.name_scope(self.name):
                with tf.name_scope('conv_' + str(self.number_of_layers)):
                    in_depth = self.layers[-1].get_shape()[-1].value
                    F = tf.Variable(initial_value=tf.truncated_normal(shape=kernel_size + [in_depth] + [out_depth],
                                                                      stddev=10. / out_depth),
                                    dtype=tf.float32,
                                    name='Filter_conv_' + str(self.number_of_layers))
                    h = tf.nn.conv2d(input=self.layers[-1],
                                     filter=F,
                                     strides=stride,
                                     padding=padding)
                    b = tf.Variable(initial_value=tf.zeros(h.get_shape()[1:]),
                                    dtype=tf.float32,
                                    name='Bias_conv_' + str(self.number_of_layers))

                    if (relu):
                        h = tf.nn.relu(tf.nn.conv2d(input=self.layers[-1],
                                                    filter=F,
                                                    strides=stride,
                                                    padding=padding) + b,
                                       name='Conv_' + str(self.number_of_layers))
                    else:
                        h = tf.add(tf.nn.conv2d(input=self.layers[-1],
                                                filter=F,
                                                strides=stride,
                                                padding=padding),
                                   b,
                                   name='Conv_' + str(self.number_of_layers))
                    self.variables.append(F)
                    self.variables.append(b)
                    self.layers.append(h)
                self.number_of_layers += 1

        def add_MaxPool_Layer(self, factor=2):
           
            with tf.name_scope(self.name):
                with tf.name_scope('Max_Pool_' + str(self.number_of_layers)):
                    self.last_layer = tf.nn.max_pool(self.last_layer,
                                                     ksize=[1, 2, 2, 1],
                                                     strides=[1, 2, 2, 1],
                                                     name='Max_Pool_Layer_' + str(self.number_of_layers),
                                                     padding='SAME')

        def add_deconv_Layer(self, out_depth, relu=True):
           
            with tf.name_scope(self.name):
                with tf.name_scope('Deconv_Layer_' + str(self.number_of_layers)):
                    in_depth = self.layers[-1]._shape[-1].value
                    out_height = 2 * self.layers[-1]._shape[1].value
                    out_width = 2 * self.layers[-1]._shape[2].value
                    F = tf.Variable(initial_value=tf.truncated_normal(shape=[3, 3, out_depth, in_depth],
                                                                      stddev=0.1,
                                                                      dtype=tf.float32),
                                    name='Filter_Deconv_' + str(self.number_of_layers))
                    if (relu):
                        h = tf.nn.relu(tf.nn.conv2d_transpose(value=self.layers[-1],
                                                              filter=F,
                                                              output_shape=[self.batch_size, out_height, out_width,
                                                                            out_depth],
                                                              strides=[1, 2, 2, 1],
                                                              padding="SAME"),
                                       name='Deconv_Layer_' + str(self.number_of_layers))
                    else:
                        h = tf.nn.conv2d_transpose(value=self.layers[-1],
                                                   filter=F,
                                                   output_shape=[self.batch_size, out_height, out_width, out_depth],
                                                   strides=[1, 2, 2, 1],
                                                   padding="SAME",
                                                   name='Deconv_Layer_' + str(self.number_of_layers))
                    self.variables.append(F)
                    self.layers.append(h)
                self.number_of_layers += 1
    """
    def compute_output(self, top1=True):
        """
            Computes the output of the network.
            @ args :
                - top1 : indicates whether to keep only the top1 prediction or not.
            @ does :
                - updates the output tensor of the network
        """
        with tf.name_scope('Output'):
            if (top1):
                self.output = tf.argmax(tf.nn.softmax(self.last_layer),
                                        axis=len(self.last_layer.get_shape()[1:].as_list()),
                                        name='output')
            else:
                self.output = tf.nn.softmax(logits=self.last_layer,
                                            name='output')

    def add_complete_encoding_layer(self, depth, layerindex, bias=True, num_conv=2, ksize=[3, 3], pooling=True,
                                    relu=True, monitor=False):
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
        with tf.name_scope('Variables_Encoding_' + str(layerindex)):
            F = []
            if (bias):
                B = []
            for i in range(num_conv):
                new_filter = tf.Variable(
                    initial_value=tf.truncated_normal(shape=[ksize[0], ksize[1], in_shape[-1], depth],
                                                      stddev=0.01,
                                                      dtype=tf.float32,
                                                      name='Filter_' + str(layerindex) + '_' + str(i)
                                                      )
                )
                if (monitor):
                    with tf.name_scope('_Filter_' + str(i)):
                        helpers.variable_summaries(new_filter)
                F.append(new_filter)
                if (bias):
                    new_bias = tf.Variable(initial_value=tf.random_uniform(minval=-1, maxval=1, shape=[depth]),
                                           dtype=tf.float32,
                                           name='Bias_' + str(layerindex) + '_' + str(i))
                    if (monitor):
                        with tf.name_scope('_Bias_' + str(i)):
                            helpers.variable_summaries(new_bias)
                    B.append(new_bias)
                in_shape[-1] = depth
        with tf.name_scope('Encoding_' + str(layerindex)):
            self.encoding_layers.append(self.last_layer)
            for i in range(num_conv):
                in_shape = self.last_layer.get_shape()[1:].as_list()
                if (bias):
                    if (relu):
                        self.last_layer = tf.nn.relu(
                            tf.nn.conv2d(self.last_layer, F[i], strides=[1, 1, 1, 1], padding="SAME") + B[i])
                    else:
                        self.last_layer = tf.nn.conv2d(self.last_layer, F[i], strides=[1, 1, 1, 1], padding="SAME") + B[
                            i]
                else:
                    if (relu):
                        self.last_layer = tf.nn.relu(
                            tf.nn.conv2d(self.last_layer, F[i], strides=[1, 1, 1, 1], padding="SAME"))
                    else:
                        self.last_layer = tf.nn.conv2d(self.last_layer, F[i], strides=[1, 1, 1, 1], padding="SAME")
                if (bias):
                    self.encoder_variables.extend(B + F)
                else:
                    self.encoder_variables.extend(F)
                self.last_layer = tf.layers.batch_normalization(self.last_layer)
            if (pooling):
                self.last_layer = tf.nn.max_pool(self.last_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                 padding="SAME", name='pooling')
                self.last_layer = tf.layers.batch_normalization(self.last_layer)

    def add_complete_decoding_layer(self, corresponding_encoding, bias=False, num_conv=0, ksize=3, init_weight=0.5):
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
        with tf.name_scope('Variables_Decoding_' + str(corresponding_encoding)):
            F = []
            deconv_filter = tf.Variable(initial_value=tf.truncated_normal(
                shape=[1, 1, self.encoding_layers[corresponding_encoding].get_shape()[-1].value, depth],
                stddev=10. / depth),
                dtype=tf.float32,
                name='Unpooling_Filter')
            deconv_weight = tf.Variable(
                initial_value=init_weight * tf.ones(shape=[2 * in_shape[1], 2 * in_shape[2], depth]),
                dtype=tf.float32,
                name='Unpooling_Weight')
            # helpers.variable_summaries(deconv_weights)
            if (bias):
                B = []
            for i in range(num_conv):
                F.append(tf.Variable(initial_value=tf.truncated_normal(shape=[ksize, ksize, depth, depth],
                                                                       stddev=1. / depth,
                                                                       dtype=tf.float32,
                                                                       name='Filter_Deocde' + str(
                                                                           corresponding_encoding) + '_' + str(i)
                                                                       )
                                     )
                         )
                if (bias):
                    B.append(tf.Variable(initial_value=tf.random_uniform([depth],
                                                                         minval=-1,
                                                                         maxval=1),
                                         dtype=tf.float32,
                                         name='Bias_' + str(corresponding_encoding) + '_' + str(i))
                             )
        with tf.name_scope('Decoding_' + str(corresponding_encoding)):
            self.last_layer = tf.image.resize_bilinear(self.last_layer, size=[2 * in_shape[1], 2 * in_shape[2]])
            self.last_layer = tf.nn.l2_normalize(self.last_layer, dim=-1)
            self.encoding_layers[corresponding_encoding] = tf.nn.conv2d(self.encoding_layers[corresponding_encoding],
                                                                        deconv_filter, strides=[1, 1, 1, 1],
                                                                        padding="SAME")
            self.encoding_layers[corresponding_encoding] = tf.nn.l2_normalize(
                self.encoding_layers[corresponding_encoding], dim=-1)
            self.last_layer = tf.add(self.last_layer,
                                     tf.multiply(self.encoding_layers[corresponding_encoding], deconv_weight))
            self.decoder_variables.append(deconv_weight)
            for i in range(num_conv):
                if (bias):
                    self.last_layer = tf.nn.relu(
                        tf.nn.conv2d(self.last_layer, F[i], strides=[1, 1, 1, 1], padding="SAME") + B[i])
                else:
                    self.last_layer = tf.nn.relu(
                        tf.nn.conv2d(self.last_layer, F[i], strides=[1, 1, 1, 1], padding="SAME"))
                if (bias):
                    self.decoder_variables.extend(B + F)
                else:
                    self.decoder_variables.extend(F)
                self.last_layer = tf.layers.batch_normalization(self.last_layer)


"""
    Grap builders
    The following methods are used to build a network, and can be passed as arguments for the model building method.
    @ args :
        - input : a 4D tensor, the input of the network
        - num_lab : the number of labels the network will have to classify
    @ returns :
        -net : a network object, with all fields completed, as according to the selected architecture.
"""

def build_CNN(input,num_lab):
    """
        Builds a fully convolutionnal neural network.
        @ args :
            - graph : the tf.Graph containing the net operations
            - input : the input tensor
        @ returns :
            - net : a Network object containing the net, as described in the paper by J.Long et al.
    """
    with tf.name_scope('CNN'):
        net = Network(input,numlab=numlab)
        net.add_complete_encoding_layer(depth=32, layerindex=0, pooling=True, bias=True, num_conv=3)
        net.add_complete_encoding_layer(depth=64, layerindex=1, pooling=True, bias=True, num_conv=3, monitor=False)
        net.add_complete_encoding_layer(depth=128, layerindex=2, num_conv=2, pooling=True, bias=True)
        net.add_complete_encoding_layer(depth=256, layerindex=3, num_conv=2, pooling=True, bias=True)
        #net.add_complete_encoding_layer(depth=512, layerindex=4, num_conv=2, pooling=True, bias=True)
        # net.add_complete_encoding_layer(depth = 512, layerindex=5,num_conv = 2, pooling = True, bias = True)
        #net.add_complete_encoding_layer(depth=256, layerindex=4, num_conv=1, pooling=False, bias=False, relu=True,
                                        #ksize=[8, 4])
        # net.add_complete_decoding_layer(corresponding_encoding=6,num_conv=0,bias = False)
        net.add_complete_decoding_layer(corresponding_encoding=3, bias=True, num_conv=1, init_weight=0.5)
        # net.add_complete_encoding_layer(depth=128,layerindex=10,num_conv=1,bias=False,pooling=False)
        # net.add_complete_decoding_layer(corresponding_encoding=4,bias=True,num_conv=1,init_weight=0.5)
        net.add_complete_decoding_layer(corresponding_encoding=2, bias=True, num_conv=1, init_weight=0.5)
        #net.add_complete_decoding_layer(corresponding_encoding=2, bias=False, num_conv=1, init_weight=0.5)
        net.add_complete_decoding_layer(corresponding_encoding=1, bias=False, num_conv=1, init_weight=0.5)
        net.add_complete_decoding_layer(corresponding_encoding=0, bias=False, num_conv=1, init_weight=0.5)
        net.add_complete_encoding_layer(depth=net.numlabs, layerindex=7, num_conv=1, pooling=False, bias=False, ksize=[1,1],
                                        relu=False)
        net.compute_output(top1=True)
    return net

def build_graph(input,numlab):
    net = Network(input,numlab=numlab)
    with tf.name_scope('Inputs'):
        input_shape = input.get_shape().as_list()

    with tf.name_scope('Conv_1_Pool'):
        W_conv11 = tf.Variable(
            initial_value=tf.truncated_normal(shape=[5, 5, input_shape[-1], 32], stddev=0.1, dtype=tf.float32)
            )
        b_conv11 = tf.Variable(initial_value=0. * tf.ones(shape=[32]), dtype=tf.float32)
        h_conv11 = tf.nn.relu(tf.nn.conv2d(input, W_conv11, strides=[1, 1, 1, 1], padding="SAME") + b_conv11,
                              name="Conv1")

        W_conv12 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1, dtype=tf.float32)
                               )
        b_conv12 = tf.Variable(initial_value=0. * tf.ones(shape=[64]), dtype=tf.float32)
        h_conv12 = tf.nn.relu(tf.nn.conv2d(h_conv11, W_conv12, strides=[1, 1, 1, 1], padding="SAME") + b_conv12,
                              name="Conv2")

        h_pool1 = tf.nn.max_pool(h_conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="Pooling")

    with tf.name_scope("Conv_2_Pool"):
        W_conv21 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 64, 128], stddev=0.1, dtype=tf.float32)
                               )
        b_conv21 = tf.Variable(initial_value=0. * tf.ones(shape=[128]), dtype=tf.float32)
        h_conv21 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv21, strides=[1, 1, 1, 1], padding="SAME") + b_conv21,
                              name="Conv1")

        W_conv22 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 128, 256], stddev=0.1, dtype=tf.float32)
                               )
        b_conv22 = tf.Variable(initial_value=0. * tf.ones(shape=[256]), dtype=tf.float32)
        h_conv22 = tf.nn.relu(tf.nn.conv2d(h_conv21, W_conv22, strides=[1, 1, 1, 1], padding="SAME") + b_conv22,
                              name="Conv2")

        h_pool2 = tf.nn.max_pool(h_conv22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="Pooling")

    with tf.name_scope('Conv_3'):
        W_conv31 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 256, 256], stddev=0.1, dtype=tf.float32)
                               )
        b_conv31 = tf.Variable(initial_value=0. * tf.ones(shape=[256]), dtype=tf.float32)
        h_conv31 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv31, strides=[1, 1, 1, 1], padding="SAME") + b_conv31,
                              name="Conv1")

        W_conv32 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 256, 256], stddev=0.1, dtype=tf.float32)
                               )
        b_conv32 = tf.Variable(initial_value=0. * tf.ones(shape=[256]), dtype=tf.float32)
        h_conv32 = tf.nn.relu(tf.nn.conv2d(h_conv31, W_conv32, strides=[1, 1, 1, 1], padding="SAME") + b_conv32,
                              name="Conv2")

    with tf.name_scope('Upscaling_1'):
        h_up4 = tf.image.resize_bilinear(h_conv32, size=[int(input_shape[1] / 2), int(input_shape[2] / 2)])
        h_merged4 = tf.concat([h_conv22, h_up4], axis=-1)
    with tf.name_scope('Conv_4'):
        W_conv41 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 512, 256], stddev=0.1, dtype=tf.float32)
                               )
        b_conv41 = tf.Variable(initial_value=0. * tf.ones(shape=[256]), dtype=tf.float32)
        h_conv41 = tf.nn.relu(tf.nn.conv2d(h_merged4, W_conv41, strides=[1, 1, 1, 1], padding="SAME") + b_conv41,
                              name="Conv1")

        W_conv42 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 256, 256], stddev=0.1, dtype=tf.float32)
                               )
        b_conv42 = tf.Variable(initial_value=0. * tf.ones(shape=[256]), dtype=tf.float32)
        h_conv42 = tf.nn.relu(tf.nn.conv2d(h_conv41, W_conv42, strides=[1, 1, 1, 1], padding="SAME") + b_conv42,
                              name="Conv2")

    with tf.name_scope('Upscaling_2'):
        h_up5 = tf.image.resize_bilinear(h_conv42, size=[int(input_shape[1]), int(input_shape[2])])
        h_merged5 = tf.concat([h_conv12, h_up5], axis=-1)

    with tf.name_scope('Conv_5'):
        W_conv51 = tf.Variable(
            initial_value=tf.truncated_normal(shape=[5, 5, 256 + 64, 256], stddev=0.1, dtype=tf.float32)
            )
        b_conv51 = tf.Variable(initial_value=0. * tf.ones(shape=[256]), dtype=tf.float32)
        h_conv51 = tf.nn.relu(tf.nn.conv2d(h_merged5, W_conv51, strides=[1, 1, 1, 1], padding="SAME") + b_conv51,
                              name="Conv1")

        W_conv52 = tf.Variable(
            initial_value=tf.truncated_normal(shape=[3, 3, 256, net.numlabs], stddev=0.1, dtype=tf.float32)
            )
        b_conv52 = tf.Variable(initial_value=0. * tf.ones(shape=[net.numlabs]), dtype=tf.float32)
        output = tf.add(tf.nn.conv2d(h_conv51, W_conv52, strides=[1, 1, 1, 1], padding="SAME"), b_conv52, name="Conv2")

    with tf.name_scope('Output'):
        net.last_layer = output
        net.compute_output(top1=True)

    return net

def build_little_CNN_1conv(input,numlab):
    with tf.name_scope('Little_Net'):
        net = Network(input,numlab=numlab)
        conv1 = tf.layers.conv2d(inputs=net.input,filters=net.numlabs,kernel_size=[3,3],activation=None,padding='SAME')
        #conv2 = tf.layers.conv2d(inputs=conv1,filters=helper.num_labels,kernel_size=[3,3],activation=None,padding='SAME')
        net.last_layer = conv1
        net.compute_output()
    return net

def build_little_CNN_2conv(input,numlab):
    with tf.name_scope('Little_Net'):
        net = Network(input, numlab=numlab)
        conv1 = tf.layers.conv2d(inputs = net.input,
                                 filters = 32,
                                 kernel_size = [3,3],
                                 padding = 'SAME',
                                 activation = tf.nn.relu
                                 )
        conv2 = tf.layers.conv2d(inputs = conv1,
                                 filters = helper.num_labels,
                                 kernel_size = [3,3],
                                 padding = 'SAME',
                                 activation = None
                                 )
        net.last_layer = conv2
        net.compute_output()
    return net

def build_little_CNN_pool_unpool(input,numlab):
    with tf.name_scope('Med_net'):
        net = Network(input,numlab=numlab)
        conv1 = tf.layers.conv2d(inputs = net.input,
                                 filters = 64,
                                 kernel_size = [3,3],
                                 padding = 'SAME',
                                 activation = tf.nn.relu
                                 )
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        with tf.name_scope('Pool1'):
            helpers.image_summaries(tf.expand_dims(pool1[:,:,:,0],axis=-1))
        unpool1 = tf.layers.conv2d_transpose(inputs=pool1,
                                             filters = helper.num_labels,
                                             kernel_size = [3,3],
                                             strides = (2,2),
                                             padding = 'SAME'
                                             )
        net.last_layer = unpool1
        net.compute_output()
    return net

def build_little_CNN_2conv_pool_2conv_pool_unpool_unpool(input,numlab):
    net = Network(input,numlab=numlab)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,_ = helpers.conv2d(input = net.input,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
        with tf.name_scope('_2'):
            conv1_2,_ = helpers.conv2d(input = conv1_1,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        helpers.image_summaries(tf.expand_dims(pool1[:,:,:,0],axis=-1))
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 128,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2,_ = helpers.conv2d(input = conv2_1,
                                       filters = 128,
                                       layername = 'Conv2_2'
                                       )
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )

    with tf.name_scope('Unpooling'):
        unpool2,unpool2vars = helpers.conv2d_transpose(pool2,
                                                       64,ksize=[3,3],
                                                       layername='unpool2'
                                                       )
        unpool1,unpool1vars = helpers.conv2d_transpose(unpool2,
                                                       helper.num_labels,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = False
                                                       )
    net.last_layer = unpool1
    net.compute_output()
    return net

def build_little_CNN_3conv_pool_2conv_pool_3conv_unpool_unpool(input,numlab):
    net = Network(input,numlab=numlab)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,var1_1 = helpers.conv2d(input = net.input,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(var1_1[0])
        with tf.name_scope('_2'):
            conv1_2,_ = helpers.conv2d(input = conv1_1,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 128,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2,_ = helpers.conv2d(input = conv2_1,
                                       filters = 128,
                                       layername = 'Conv2_2'
                                       )
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_3'):
        with tf.name_scope('_1'):
            conv3_1, var3_1 = helpers.conv2d(input=pool2,
                                        layername = 'Conv3_1',
                                        filters = 256
                                        )
            helpers.variable_summaries(var3_1[0])
        with tf.name_scope('_2'):
            conv3_2, _ = helpers.conv2d(input=conv3_1,
                                     filters = 256,
                                     layername = 'Conv3_2'
                                     )
        with tf.name_scope('_3'):
            conv3_3, _ = helpers.conv2d(input=conv3_2,
                                     filters = 256,
                                     layername = 'Conv3_3',
                                     ksize = [1,1]
                                     )

    with tf.name_scope('Unpooling'):
        unpool2,unpool2vars = helpers.conv2d_transpose(conv3_3,
                                                       64,ksize=[3,3],
                                                       layername='unpool2'
                                                       )
        unpool1,unpool1vars = helpers.conv2d_transpose(unpool2,
                                                       helper.num_labels,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = False
                                                       )
    net.last_layer = unpool1
    net.compute_output()
    return net

def build_little_CNN_3conv_pool_2conv_pool_3conv_unpool_merge_unpool(input,numlab):
    net = Network(input,numlab=numlab)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,var1_1 = helpers.conv2d(input = net.input,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(var1_1[0])
        with tf.name_scope('_2'):
            conv1_2,_ = helpers.conv2d(input = conv1_1,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 128,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2,_ = helpers.conv2d(input = conv2_1,
                                       filters = 128,
                                       layername = 'Conv2_2'
                                       )
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_3'):
        with tf.name_scope('_1'):
            conv3_1, var3_1 = helpers.conv2d(input=pool2,
                                        layername = 'Conv3_1',
                                        filters = 256
                                        )
            helpers.variable_summaries(var3_1[0])
        with tf.name_scope('_2'):
            conv3_2, _ = helpers.conv2d(input=conv3_1,
                                     filters = 256,
                                     layername = 'Conv3_2'
                                     )
        with tf.name_scope('_3'):
            conv3_3, _ = helpers.conv2d(input=conv3_2,
                                     filters = 256,
                                     layername = 'Conv3_3',
                                     ksize = [1,1]
                                     )

    with tf.name_scope('Unpooling_1'):
        unpool1,unpool1vars = helpers.conv2d_transpose(conv3_3,
                                                       64,ksize=[3,3],
                                                       layername='unpool2'
                                                       )
    with tf.name_scope('Merge'):
        merged, mergedvars = helpers.merge_layers(largelayer = unpool1,
                                                  smalllayer = pool2,
                                                  num_of_ups = 1)
    with tf.name_scope('Unpooling_2'):
        unpool2,unpool2vars = helpers.conv2d_transpose(merged,
                                                       helper.num_labels,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = False
                                                       )
    net.last_layer = unpool2
    net.compute_output()
    return net

def build_little_CNN_3conv_pool_2conv_pool_3conv_pool_3conv_unpool_merge_unpool_unpool(input,numlab):
    net = Network(input,numlab=numlab)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,var1_1 = helpers.conv2d(input = net.input,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(var1_1[0])
        with tf.name_scope('_2'):
            conv1_2,_ = helpers.conv2d(input = conv1_1,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 128,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2,_ = helpers.conv2d(input = conv2_1,
                                       filters = 128,
                                       layername = 'Conv2_2'
                                       )
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_3'):
        with tf.name_scope('_1'):
            conv3_1, var3_1 = helpers.conv2d(input=pool2,
                                        layername = 'Conv3_1',
                                        filters = 256
                                        )
            helpers.variable_summaries(var3_1[0])
        with tf.name_scope('_2'):
            conv3_2, _ = helpers.conv2d(input=conv3_1,
                                     filters = 256,
                                     layername = 'Conv3_2'
                                     )
        with tf.name_scope('_3'):
            conv3_3, _ = helpers.conv2d(input=conv3_2,
                                     filters = 256,
                                     layername = 'Conv3_3',
                                     ksize = [1,1]
                                     )
    with tf.name_scope('Pool3'):
         pool3 = tf.layers.max_pooling2d(inputs=conv3_3,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_4'):
         with tf.name_scope('_1'):
            conv4_1, var3_1 = helpers.conv2d(input=pool3,
                                        layername = 'Conv4_1',
                                        filters = 512
                                        )
            helpers.variable_summaries(var3_1[0])
         with tf.name_scope('_2'):
            conv4_2, _ = helpers.conv2d(input=conv4_1,
                                     filters = 512,
                                     layername = 'Conv4_2'
                                     )
         with tf.name_scope('_3'):
            conv4_3, _ = helpers.conv2d(input=conv4_2,
                                     filters = 512,
                                     layername = 'Conv4_3',
                                     ksize = [1,1]
                                     )
    with tf.name_scope('Unpooling_1'):
        unpool1,unpool1vars = helpers.conv2d_transpose(conv4_3,
                                                       helper.num_labels,
                                                       ksize=[3,3],
                                                       layername='unpool2'
                                                       )
    with tf.name_scope('Merge'):
        merged, mergedvars = helpers.merge_layers(largelayer = unpool1,
                                                  smalllayer = pool3,
                                                  num_of_ups = 1)
    with tf.name_scope('Unpooling_2'):
        unpool2,unpool2vars = helpers.conv2d_transpose(merged,
                                                       helper.num_labels,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = True
                                                       )
    with tf.name_scope('Unpooling_3'):
        unpool3, unpool3vars = helpers.conv2d_transpose(unpool2,
                                                        helper.num_labels,
                                                        ksize=[3,3],
                                                        layername = 'unpool3',
                                                        relu = False
                                                        )
    net.last_layer = unpool3
    net.compute_output()
    return net

def build_little_CNN_2skips(input,numlab):
    net = Network(input,numlab=numlab)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,var1_1 = helpers.conv2d(input = net.input,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(var1_1[0])
        with tf.name_scope('_2'):
            conv1_2,_ = helpers.conv2d(input = conv1_1,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 128,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2,_ = helpers.conv2d(input = conv2_1,
                                       filters = 128,
                                       layername = 'Conv2_2'
                                       )
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_3'):
        with tf.name_scope('_1'):
            conv3_1, var3_1 = helpers.conv2d(input=pool2,
                                        layername = 'Conv3_1',
                                        filters = 256
                                        )
            helpers.variable_summaries(var3_1[0])
        with tf.name_scope('_2'):
            conv3_2, _ = helpers.conv2d(input=conv3_1,
                                     filters = 256,
                                     layername = 'Conv3_2'
                                     )
        with tf.name_scope('_3'):
            conv3_3, _ = helpers.conv2d(input=conv3_2,
                                     filters = 256,
                                     layername = 'Conv3_3',
                                     ksize = [1,1]
                                     )
    with tf.name_scope('Pool3'):
         pool3 = tf.layers.max_pooling2d(inputs=conv3_3,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_4'):
         with tf.name_scope('_1'):
            conv4_1, var3_1 = helpers.conv2d(input=pool3,
                                        layername = 'Conv4_1',
                                        filters = 512
                                        )
            helpers.variable_summaries(var3_1[0])
         with tf.name_scope('_2'):
            conv4_2, _ = helpers.conv2d(input=conv4_1,
                                     filters = 512,
                                     layername = 'Conv4_2'
                                     )
         with tf.name_scope('_3'):
            conv4_3, _ = helpers.conv2d(input=conv4_2,
                                     filters = 512,
                                     layername = 'Conv4_3',
                                     ksize = [1,1]
                                     )
    with tf.name_scope('Unpooling_4'):
        unpool4_1,unpool4_1vars = helpers.conv2d_transpose(conv4_3,
                                                           filters = net.numlabs,
                                                           ksize=[3,3],
                                                           layername='unpool4_1'
                                                           )
        unpool4_2, unpool4_2vars = helpers.conv2d_transpose(unpool4_1,
                                                            filters = net.numlabs,
                                                            ksize = [3,3],
                                                            layername = 'Unpool4_2'
                                                            )
    with tf.name_scope('Unpooling_3'):
        unpool_3, unpool_3vars = helpers.conv2d_transpose(pool2,
                                                        filters=net.numlabs,
                                                        layername = 'Unpooling_3',
                                                        ksize = [3,3]
                                                        )
    with tf.name_scope('Merge'):
        with tf.name_scope('Predictions'):
            pred4 = helpers.predictions(unpool4_2,numlabs=net.numlabs)
            pred3 = helpers.predictions(unpool_3,numlabs=net.numlabs)
            pred2 = helpers.predictions(pool1,numlabs=net.numlabs)
        with tf.name_scope('Merging'):
            #merged,_ = helpers.alternate_merge([pred4,pred3,pred2],
            #                                  ksize = [3,3]
            #                                  )
            merged = tf.add(pred4,tf.add(pred2,pred3))
    with tf.name_scope('Unpooling_2'):
        unpool2,unpool2vars = helpers.conv2d_transpose(merged,
                                                       net.numlabs,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = False
                                                       )
    
    net.last_layer = unpool2
    net.compute_output()
    return net

def build_little_CNN_2skips_bilinupsambling(input,numlab):
    net = Network(input,numlab=numlab)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,var1_1 = helpers.conv2d(input = net.input,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(var1_1[0])
        with tf.name_scope('_2'):
            conv1_2,_ = helpers.conv2d(input = conv1_1,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 128,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2,_ = helpers.conv2d(input = conv2_1,
                                       filters = 128,
                                       layername = 'Conv2_2'
                                       )
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_3'):
        with tf.name_scope('_1'):
            conv3_1, var3_1 = helpers.conv2d(input=pool2,
                                        layername = 'Conv3_1',
                                        filters = 256
                                        )
            helpers.variable_summaries(var3_1[0])
        with tf.name_scope('_2'):
            conv3_2, _ = helpers.conv2d(input=conv3_1,
                                     filters = 256,
                                     layername = 'Conv3_2'
                                     )
        with tf.name_scope('_3'):
            conv3_3, _ = helpers.conv2d(input=conv3_2,
                                     filters = 256,
                                     layername = 'Conv3_3',
                                     ksize = [1,1]
                                     )
    with tf.name_scope('Pool3'):
         pool3 = tf.layers.max_pooling2d(inputs=conv3_3,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_4'):
         with tf.name_scope('_1'):
            conv4_1, var3_1 = helpers.conv2d(input=pool3,
                                        layername = 'Conv4_1',
                                        filters = 512
                                        )
            helpers.variable_summaries(var3_1[0])
         with tf.name_scope('_2'):
            conv4_2, _ = helpers.conv2d(input=conv4_1,
                                     filters = 512,
                                     layername = 'Conv4_2'
                                     )
         with tf.name_scope('_3'):
            conv4_3, _ = helpers.conv2d(input=conv4_2,
                                     filters = 512,
                                     layername = 'Conv4_3',
                                     ksize = [1,1]
                                     )
    with tf.name_scope('Unpooling_4'):
        #unpool4_1,unpool4_1vars = helpers.conv2d_transpose(conv4_3,
        #                                                   filters = net.numlabs,
        #                                                   ksize=[3,3],
        #                                                   layername='unpool4_1'
        #                                                   )
        #unpool4_2, unpool4_2vars = helpers.conv2d_transpose(unpool4_1,
        #                                                    filters = net.numlabs,
        #                                                    ksize = [3,3],
        #                                                    layername = 'Unpool4_2'
        #                                                    )
        unpool4_2 = tf.image.resize_bilinear(conv4_3,
                                             size = [4*conv4_3.get_shape()[1].value, 4 * conv4_3.get_shape()[2].value]
                                             )
    with tf.name_scope('Unpooling_3'):
        unpool_3 = tf.image.resize_bilinear(pool2,
                                             size = [2*pool2.get_shape()[1].value, 2 * pool2.get_shape()[2].value]
                                             )
    with tf.name_scope('Merge'):
        with tf.name_scope('Predictions'):
            pred4 = helpers.predictions(unpool4_2,numlabs=net.numlabs)
            pred3 = helpers.predictions(unpool_3,numlabs=net.numlabs)
            pred2 = helpers.predictions(pool1,numlabs=net.numlabs)
        with tf.name_scope('Merging'):
            #merged,_ = helpers.alternate_merge([pred4,pred3,pred2],
            #                                   numlabs = net.numlabs,
            #                                  ksize = [3,3]
            #                                  )
            merged = tf.add(pred4,tf.add(pred2,pred3))
    with tf.name_scope('Unpooling_2'):
        #unpool2,unpool2vars = helpers.conv2d_transpose(merged,
        #                                               net.numlabs,
        #                                               ksize=[3,3],
        #                                               layername='unpool2',
        #                                               relu = False
        #                                               )
        unpool2 = tf.image.resize_bilinear(merged,
                                           size = [2*merged.get_shape()[1].value, 2*merged.get_shape()[2].value]
                                           )
    #with tf.name_scope('Unpooling_3'):
    #    unpool3, unpool3vars = helpers.conv2d_transpose(unpool2,
    #                                                    helper.num_labels,
    #                                                    ksize=[3,3],
    #                                                    layername = 'unpool3',
    #                                                    relu = False
    #                                                    )
    net.last_layer = unpool2
    net.compute_output()
    return net

def build_big_CNN_2skips_bilinupsambling(input,numlab):
    net = Network(input,numlab=numlab)

    with tf.name_scope('Conv0'):
        with tf.name_scope('_1'):
            conv0_1, _ = helpers.conv2d(input = net.input,
                                        filters = 32,
                                        layername='Conv0_1'
                                        )
        with tf.name_scope('_2'):
            conv0_2, _ = helpers.conv2d(input = conv0_1,
                                        filters = 64,
                                        layername='Conv0_1'
                                        )
    with tf.name_scope('Pool0'):
        pool0 = tf.layers.max_pooling2d(inputs=conv0_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv0bis'):
        with tf.name_scope('_1'):
            conv0bis_1, _ = helpers.conv2d(input = pool0,
                                        filters = 128,
                                        layername='Conv0bis_1'
                                        )
        with tf.name_scope('_2'):
            conv0bis_2, _ = helpers.conv2d(input = conv0bis_1,
                                        filters = 128,
                                        layername='Conv0bis_1'
                                        )
    with tf.name_scope('Pool0bis'):
        pool0bis = tf.layers.max_pooling2d(inputs=conv0bis_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,var1_1 = helpers.conv2d(input = pool0bis,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(var1_1[0])
        with tf.name_scope('_2'):
            conv1_2,_ = helpers.conv2d(input = conv1_1,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 128,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2,_ = helpers.conv2d(input = conv2_1,
                                       filters = 128,
                                       layername = 'Conv2_2'
                                       )
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_3'):
        with tf.name_scope('_1'):
            conv3_1, var3_1 = helpers.conv2d(input=pool2,
                                        layername = 'Conv3_1',
                                        filters = 256
                                        )
            helpers.variable_summaries(var3_1[0])
        with tf.name_scope('_2'):
            conv3_2, _ = helpers.conv2d(input=conv3_1,
                                     filters = 256,
                                     layername = 'Conv3_2'
                                     )
        with tf.name_scope('_3'):
            conv3_3, _ = helpers.conv2d(input=conv3_2,
                                     filters = 256,
                                     layername = 'Conv3_3',
                                     ksize = [1,1]
                                     )
    with tf.name_scope('Pool3'):
         pool3 = tf.layers.max_pooling2d(inputs=conv3_3,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_4'):
         with tf.name_scope('_1'):
            conv4_1, var3_1 = helpers.conv2d(input=pool3,
                                        layername = 'Conv4_1',
                                        filters = 512
                                        )
            helpers.variable_summaries(var3_1[0])
         with tf.name_scope('_2'):
            conv4_2, _ = helpers.conv2d(input=conv4_1,
                                     filters = 512,
                                     layername = 'Conv4_2'
                                     )
         with tf.name_scope('_3'):
            conv4_3, _ = helpers.conv2d(input=conv4_2,
                                     filters = 512,
                                     layername = 'Conv4_3',
                                     ksize = [1,1]
                                     )
    with tf.name_scope('Unpooling_4'):
        #unpool4_1,unpool4_1vars = helpers.conv2d_transpose(conv4_3,
        #                                                   filters = net.numlabs,
        #                                                   ksize=[3,3],
        #                                                   layername='unpool4_1'
        #                                                   )
        #unpool4_2, unpool4_2vars = helpers.conv2d_transpose(unpool4_1,
        #                                                    filters = net.numlabs,
        #                                                    ksize = [3,3],
        #                                                    layername = 'Unpool4_2'
        #                                                    )
        unpool4_2 = tf.image.resize_bilinear(conv4_3,
                                             size = [4*conv4_3.get_shape()[1].value, 4 * conv4_3.get_shape()[2].value]
                                             )
    with tf.name_scope('Unpooling_3'):
        unpool_3 = tf.image.resize_bilinear(pool2,
                                             size = [2*pool2.get_shape()[1].value, 2 * pool2.get_shape()[2].value]
                                             )
    with tf.name_scope('Merge'):
        with tf.name_scope('Predictions'):
            pred4 = helpers.predictions(unpool4_2,numlabs=net.numlabs)
            pred3 = helpers.predictions(unpool_3,numlabs=net.numlabs)
            pred2 = helpers.predictions(pool1,numlabs=net.numlabs)
        with tf.name_scope('Merging'):
            #merged,_ = helpers.alternate_merge([pred4,pred3,pred2],
            #                                   numlabs = net.numlabs,
            #                                  ksize = [3,3]
            #                                  )
            merged = tf.add(pred4,tf.add(pred2,pred3))
    with tf.name_scope('Unpooling_2'):
        #unpool2,unpool2vars = helpers.conv2d_transpose(merged,
        #                                               net.numlabs,
        #                                               ksize=[3,3],
        #                                               layername='unpool2',
        #                                               relu = False
        #                                               )
        unpool2 = tf.image.resize_bilinear(merged,
                                           size = [8*merged.get_shape()[1].value, 8*merged.get_shape()[2].value]
                                           )
    #with tf.name_scope('Unpooling_3'):
    #    unpool3, unpool3vars = helpers.conv2d_transpose(unpool2,
    #                                                    helper.num_labels,
    #                                                    ksize=[3,3],
    #                                                    layername = 'unpool3',
    #                                                    relu = False
    #                                                    )
    net.last_layer = unpool2
    net.compute_output()
    return net

def build_big_CNN_2skips(input,numlab):
    net = Network(input,numlab=numlab)
    net.variables['32s'] = []
    net.variables['16s'] = []
    net.variables['8s'] = []

    with tf.name_scope('Conv0'):
        with tf.name_scope('_1'):
            conv0_1, conv0_1vars = helpers.conv2d(input = net.input,
                                        filters = 32,
                                        layername='Conv0_1'
                                        )
        with tf.name_scope('_2'):
            conv0_2, conv0_2vars = helpers.conv2d(input = conv0_1,
                                        filters = 64,
                                        layername='Conv0_1'
                                        )
            net.variables['32s'].extend(conv0_1vars)
            net.variables['32s'].extend(conv0_2vars)
    with tf.name_scope('Pool0'):
        pool0 = tf.layers.max_pooling2d(inputs=conv0_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv0bis'):
        with tf.name_scope('_1'):
            conv0bis_1, conv0bis_1vars = helpers.conv2d(input = pool0,
                                        filters = 128,
                                        layername='Conv0bis_1'
                                        )
        with tf.name_scope('_2'):
            conv0bis_2, conv0bis_2vars = helpers.conv2d(input = conv0bis_1,
                                        filters = 128,
                                        layername='Conv0bis_1'
                                        )
            net.variables['32s'].extend(conv0bis_1vars)
            net.variables['32s'].extend(conv0bis_2vars)
    with tf.name_scope('Pool0bis'):
        pool0bis = tf.layers.max_pooling2d(inputs=conv0bis_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,conv1_1vars = helpers.conv2d(input = pool0bis,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(conv1_1vars[0])
        with tf.name_scope('_2'):
            conv1_2,conv1_2vars = helpers.conv2d(input = conv1_1,
                                       filters = 64,
                                       layername = 'Conv1_1'
                                       )
            net.variables['32s'].extend(conv1_1vars)
            net.variables['32s'].extend(conv1_2vars)
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 128,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2, conv2_2vars = helpers.conv2d(input = conv2_1,
                                       filters = 128,
                                       layername = 'Conv2_2'
                                       )
            net.variables['32s'].extend(conv2_1vars)
            net.variables['32s'].extend(conv2_2vars)
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_3'):
        with tf.name_scope('_1'):
            conv3_1, conv3_1vars = helpers.conv2d(input=pool2,
                                        layername = 'Conv3_1',
                                        filters = 256
                                        )
            helpers.variable_summaries(conv3_1vars[0])
        with tf.name_scope('_2'):
            conv3_2, conv3_2vars = helpers.conv2d(input=conv3_1,
                                     filters = 256,
                                     layername = 'Conv3_2'
                                     )
        with tf.name_scope('_3'):
            conv3_3, conv3_3vars = helpers.conv2d(input=conv3_2,
                                     filters = 256,
                                     layername = 'Conv3_3',
                                     ksize = [1,1]
                                     )
            net.variables['32s'].extend(conv3_1vars)
            net.variables['32s'].extend(conv3_2vars)
            net.variables['32s'].extend(conv3_3vars)
    with tf.name_scope('Pool3'):
         pool3 = tf.layers.max_pooling2d(inputs=conv3_3,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_4'):
         with tf.name_scope('_1'):
            conv4_1, conv4_1vars = helpers.conv2d(input=pool3,
                                        layername = 'Conv4_1',
                                        filters = 512
                                        )
            helpers.variable_summaries(conv4_1vars[0])
            net.variables['32s'].extend(conv4_1vars)
         with tf.name_scope('_2'):
            conv4_2, conv4_2vars = helpers.conv2d(input=conv4_1,
                                     filters = 512,
                                     layername = 'Conv4_2'
                                     )
         with tf.name_scope('_3'):
            conv4_3, conv4_3vars = helpers.conv2d(input=conv4_2,
                                     filters = 512,
                                     layername = 'Conv4_3',
                                     ksize = [1,1]
                                     )
            net.variables['32s'].extend(conv4_2vars)
            net.variables['32s'].extend(conv4_3vars)
    with tf.name_scope('Unpooling_4'):
        unpool4_1,unpool4_1vars = helpers.conv2d_transpose(conv4_3,
                                                           filters = net.numlabs,
                                                           ksize=[3,3],
                                                           layername='unpool4_1'
                                                           )
        unpool4_2, unpool4_2vars = helpers.conv2d_transpose(unpool4_1,
                                                            filters = net.numlabs,
                                                            ksize = [3,3],
                                                            layername = 'Unpool4_2'
                                                            )
        net.variables['32s'].extend(unpool4_1vars)
        net.variables['32s'].extend(unpool4_2vars)
        #unpool4_2 = tf.image.resize_bilinear(conv4_3,
        #                                     size = [4*conv4_3.get_shape()[1].value, 4 * conv4_3.get_shape()[2].value]
        #                                     )
    with tf.name_scope('Unpooling_4bis'):
        unpool_4bis, unpool4bisvars = helpers.conv2d_transpose(pool2,
                                               filters = net.numlabs,
                                               ksize = [3,3],
                                               layername = 'Unpool_4bis',
                                               k_init = [0,0]
                                               )
        net.variables['16s'].extend(unpool4bisvars)
        #unpool_3 = tf.image.resize_bilinear(pool2,
        #                                     size = [2*pool2.get_shape()[1].value, 2 * pool2.get_shape()[2].value]
        #                                     )
    with tf.name_scope('Merge'):
        with tf.name_scope('Resizing'):
            #pred4, _ = helpers.conv2d(input=unpool4_2, layername = 'Reshape1',ksize=[1,1],filters=net.numlabs)
            #pred3, _ = helpers.conv2d(input=unpool3, layername = 'Reshape2',ksize=[1,1],filters=net.numlabs)
            pred2, pred2vars = helpers.conv2d(input=pool1, 
                                              layername = 'Reshape3',
                                              ksize=[1,1],
                                              filters=net.numlabs,
                                              k_init = [0,0]
                                              )
            net.variables['8s'].extend(pred2vars)
        with tf.name_scope('Merging'):
            #merged,_ = helpers.alternate_merge([pred4,pred3,pred2],
            #                                   numlabs = net.numlabs,
            #                                  ksize = [3,3]
            #                                  )
            merged = tf.add(unpool4_2,tf.add(pred2,unpool_4bis))
    with tf.name_scope('Unpooling_2'):
        unpool2,unpool2vars = helpers.conv2d_transpose(merged,
                                                       net.numlabs,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = True
                                                       )
        net.variables['32s'].extend(unpool2vars)
        #unpool2 = tf.image.resize_bilinear(merged,
                                           #size = [8*merged.get_shape()[1].value, 8*merged.get_shape()[2].value]
                                           #)
    with tf.name_scope('Unpooling_3'):
        unpool3, unpool3vars = helpers.conv2d_transpose(unpool2,
                                                        net.numlabs,
                                                        ksize=[3,3],
                                                        layername = 'unpool3',
                                                        relu = True
                                                        )
        net.variables['32s'].extend(unpool3vars)
    with tf.name_scope('Unpooling_1'):
        unpool1, unpool1vars = helpers.conv2d_transpose(unpool3,
                                                        net.numlabs,
                                                        ksize=[3,3],
                                                        layername = 'unpool4',
                                                        relu = False
                                                        )
        net.variables['32s'].extend(unpool1vars)
    net.last_layer = unpool1
    net.compute_output()
    return net

def build_small_CNN_dropouts(input,numlab):
    net = Network(input,numlab=numlab)
    net.variables['32s'] = []
    net.variables['16s'] = []
    net.variables['8s'] = []

    with tf.name_scope('Conv0'):
        with tf.name_scope('_1'):
            conv0_1, conv0_1vars = helpers.conv2d(input = net.input,
                                        filters = 32,
                                        layername='Conv0_1'
                                        )
        with tf.name_scope('_2'):
            conv0_2, conv0_2vars = helpers.conv2d(input = conv0_1,
                                        filters = 64,
                                        layername='Conv0_1'
                                        )
            net.variables['32s'].extend(conv0_1vars)
            net.variables['32s'].extend(conv0_2vars)
    with tf.name_scope('Pool0'):
        pool0 = tf.layers.max_pooling2d(inputs=conv0_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool0 = tf.layers.dropout(pool0,rate = 0.5)
    with tf.name_scope('Conv0bis'):
        with tf.name_scope('_1'):
            conv0bis_1, conv0bis_1vars = helpers.conv2d(input = pool0,
                                        filters = 128,
                                        layername='Conv0bis_1'
                                        )
            net.variables['32s'].extend(conv0bis_1vars)
    with tf.name_scope('Pool0bis'):
        pool0bis = tf.layers.max_pooling2d(inputs=conv0bis_1,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool0bis = tf.layers.dropout(pool0bis, rate = 0.5)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,conv1_1vars = helpers.conv2d(input = pool0bis,
                                       filters = 256,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(conv1_1vars[0])
        with tf.name_scope('_2'):
            conv1_2,conv1_2vars = helpers.conv2d(input = conv1_1,
                                       filters = 256,
                                       layername = 'Conv1_1',
                                       ksize = [1,1]
                                       )
            net.variables['32s'].extend(conv1_1vars)
            net.variables['32s'].extend(conv1_2vars)
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool1 = tf.layers.dropout(pool1,rate=0.2)
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 256,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2, conv2_2vars = helpers.conv2d(input = conv2_1,
                                       filters = 256,
                                       layername = 'Conv2_2',
                                       ksize = [1,1]
                                       )
            net.variables['32s'].extend(conv2_1vars)
            net.variables['32s'].extend(conv2_2vars)
    with tf.name_scope('Pool2'):
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_3'):
        with tf.name_scope('_1'):
            conv3_1, conv3_1vars = helpers.conv2d(input=pool2,
                                        layername = 'Conv3_1',
                                        filters = 256
                                        )
            helpers.variable_summaries(conv3_1vars[0])
        with tf.name_scope('_2'):
            conv3_2, conv3_2vars = helpers.conv2d(input=conv3_1,
                                     filters = 256,
                                     layername = 'Conv3_2'
                                     )
        with tf.name_scope('_3'):
            conv3_3, conv3_3vars = helpers.conv2d(input=conv3_2,
                                     filters = 256,
                                     layername = 'Conv3_3',
                                     ksize = [1,1]
                                     )
            net.variables['32s'].extend(conv3_1vars)
            net.variables['32s'].extend(conv3_2vars)
            net.variables['32s'].extend(conv3_3vars)
    with tf.name_scope('Pool3'):
         pool3 = tf.layers.max_pooling2d(inputs=conv3_3,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
    with tf.name_scope('Conv_4'):
         with tf.name_scope('_1'):
            conv4_1, conv4_1vars = helpers.conv2d(input=pool3,
                                        layername = 'Conv4_1',
                                        filters = 512
                                        )
            helpers.variable_summaries(conv4_1vars[0])
            net.variables['32s'].extend(conv4_1vars)
         with tf.name_scope('_2'):
            conv4_2, conv4_2vars = helpers.conv2d(input=conv4_1,
                                     filters = 512,
                                     layername = 'Conv4_2'
                                     )
         with tf.name_scope('_3'):
            conv4_3, conv4_3vars = helpers.conv2d(input=conv4_2,
                                     filters = 512,
                                     layername = 'Conv4_3',
                                     ksize = [1,1]
                                     )
            net.variables['32s'].extend(conv4_2vars)
            net.variables['32s'].extend(conv4_3vars)
    with tf.name_scope('Unpooling_4'):
        unpool4_1,unpool4_1vars = helpers.conv2d_transpose(conv4_3,
                                                           filters = net.numlabs,
                                                           ksize=[3,3],
                                                           layername='unpool4_1'
                                                           )
        unpool4_2, unpool4_2vars = helpers.conv2d_transpose(unpool4_1,
                                                            filters = net.numlabs,
                                                            ksize = [3,3],
                                                            layername = 'Unpool4_2'
                                                            )
        net.variables['32s'].extend(unpool4_1vars)
        net.variables['32s'].extend(unpool4_2vars)
       
    with tf.name_scope('Unpooling_4bis'):
        unpool_4bis, unpool4bisvars = helpers.conv2d_transpose(pool2,
                                               filters = net.numlabs,
                                               ksize = [3,3],
                                               layername = 'Unpool_4bis',
                                               k_init = [0,0]
                                               )
        net.variables['16s'].extend(unpool4bisvars)
    with tf.name_scope('Merge'):
        with tf.name_scope('Resizing'):
            pred2, pred2vars = helpers.conv2d(input=pool1, 
                                              layername = 'Reshape3',
                                              ksize=[1,1],
                                              filters=net.numlabs,
                                              k_init = [0,0]
                                              )
            net.variables['8s'].extend(pred2vars)
        with tf.name_scope('Merging'):
            merged = tf.add(unpool4_2,tf.add(pred2,unpool_4bis))
    with tf.name_scope('Unpooling_2'):
        unpool2,unpool2vars = helpers.conv2d_transpose(merged,
                                                       net.numlabs,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = True
                                                       )
        net.variables['32s'].extend(unpool2vars)
        
    with tf.name_scope('Unpooling_3'):
        unpool3, unpool3vars = helpers.conv2d_transpose(unpool2,
                                                        net.numlabs,
                                                        ksize=[3,3],
                                                        layername = 'unpool3',
                                                        relu = True
                                                        )
        net.variables['32s'].extend(unpool3vars)
    with tf.name_scope('Unpooling_1'):
        unpool1, unpool1vars = helpers.conv2d_transpose(unpool3,
                                                        net.numlabs,
                                                        ksize=[3,3],
                                                        layername = 'unpool4',
                                                        relu = False
                                                        )
        net.variables['32s'].extend(unpool1vars)
    net.last_layer = unpool1
    net.compute_output()
    return net

def build_smallerCNN_upscaled(input,numlab):
    net = Network(input,numlab=numlab)
    net.variables['32s'] = []
    net.variables['16s'] = []
    net.variables['8s'] = []

    with tf.name_scope('Conv0'):
        with tf.name_scope('_1'):
            conv0_1, conv0_1vars = helpers.conv2d(input = net.input,
                                        filters = 32,
                                        layername='Conv0_1'
                                        )
        with tf.name_scope('_2'):
            conv0_2, conv0_2vars = helpers.conv2d(input = conv0_1,
                                        filters = 64,
                                        layername='Conv0_1'
                                        )
            net.variables['32s'].extend(conv0_1vars)
            net.variables['32s'].extend(conv0_2vars)
    with tf.name_scope('Pool0'):
        pool0 = tf.layers.max_pooling2d(inputs=conv0_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool0 = tf.layers.dropout(pool0,rate = 0.5)
    with tf.name_scope('Conv0bis'):
        with tf.name_scope('_1'):
            conv0bis_1, conv0bis_1vars = helpers.conv2d(input = pool0,
                                        filters = 128,
                                        layername='Conv0bis_1'
                                        )
            net.variables['32s'].extend(conv0bis_1vars)
    with tf.name_scope('Pool0bis'):
        pool0bis = tf.layers.max_pooling2d(inputs=conv0bis_1,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool0bis = tf.layers.dropout(pool0bis, rate = 0.5)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,conv1_1vars = helpers.conv2d(input = pool0bis,
                                       filters = 256,
                                       layername = 'Conv1_1'
                                       )
            helpers.variable_summaries(conv1_1vars[0])
        with tf.name_scope('_2'):
            conv1_2,conv1_2vars = helpers.conv2d(input = conv1_1,
                                       filters = 256,
                                       layername = 'Conv1_1',
                                       ksize = [1,1]
                                       )
            net.variables['32s'].extend(conv1_1vars)
            net.variables['32s'].extend(conv1_2vars)
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool1 = tf.layers.dropout(pool1,rate=0.2)
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d(input = pool1,
                                                  filters = 256,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2, conv2_2vars = helpers.conv2d(input = conv2_1,
                                       filters = 256,
                                       layername = 'Conv2_2',
                                       ksize = [1,1]
                                       )
            net.variables['32s'].extend(conv2_1vars)
            net.variables['32s'].extend(conv2_2vars)
    with tf.name_scope('Conv3'):
        conv3, conv3vars = helpers.conv2d_dilated(input=conv2_2,
                                                  filters = 256,
                                                  layername = 'Conv3',
                                                  factor = 2)
        net.variables['32s'].extend(conv3vars)
    with tf.name_scope('Conv4'):
        conv4, conv4vars = helpers.conv2d_dilated(input=conv3,
                                                  filters = 512,
                                                  layername = 'Conv4',
                                                  factor = 4
                                                  )
        net.variables['32s'].extend(conv4vars)
    with tf.name_scope('Merge'):
        pool1_resize, resize1vars = helpers.conv2d(input = pool1,
                                                   filters = net.numlabs,
                                                   layername = 'resizing1',
                                                   ksize = [3,3],
                                                   #k_init = [0,0]
                                                   )
        conv3_resize, resize2vars = helpers.conv2d(input = conv3,
                                                   filters = net.numlabs,
                                                   layername = 'resizing2',
                                                   ksize = [3,3],
                                                   #k_init = [0,0]
                                                   )
        conv4_resize, resize3vars = helpers.conv2d(input = conv4,
                                                   filters = net.numlabs,
                                                   layername = 'resizing3')
        helpers.variable_summaries(resize1vars[0])
        helpers.variable_summaries(resize2vars[0])
        helpers.variable_summaries(resize3vars[0])
        with tf.name_scope('pool1_resize'):
            helpers.variable_summaries(pool1_resize)
            helpers.inspect_layer(pool1_resize,3,'pool1')
        with tf.name_scope('conv3_resize'):
            helpers.variable_summaries(conv3_resize)
            helpers.inspect_layer(conv3_resize,3,'conv3')
        with tf.name_scope('conv4_resize'):
            helpers.variable_summaries(conv4_resize)
            helpers.inspect_layer(conv4_resize,3,'conv4')
        net.variables['8s'].extend(resize1vars)
        net.variables['16s'].extend(resize2vars)
        net.variables['32s'].extend(resize3vars)
        merged = tf.add(tf.add(pool1_resize,conv3_resize),conv4_resize)
    with tf.name_scope('Unpooling_2'):
        unpool2,unpool2vars = helpers.conv2d_transpose(merged,
                                                       net.numlabs,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = True
                                                       )
        net.variables['32s'].extend(unpool2vars)     
    with tf.name_scope('Unpooling_3'):
        unpool3, unpool3vars = helpers.conv2d_transpose(unpool2,
                                                        net.numlabs,
                                                        ksize=[3,3],
                                                        layername = 'unpool3',
                                                        relu = True
                                                        )
        net.variables['32s'].extend(unpool3vars)
    with tf.name_scope('Unpooling_1'):
        unpool1, unpool1vars = helpers.conv2d_transpose(unpool3,
                                                        net.numlabs,
                                                        ksize=[3,3],
                                                        layername = 'unpool4',
                                                        relu = False
                                                        )
        net.variables['32s'].extend(unpool1vars)
    net.last_layer = unpool1
    net.compute_output()
    return net

def build_smallerCNN_upscaled_only2pool(input,numlab):
    net = Network(input,numlab=numlab)
    net.variables['32s'] = []
    net.variables['16s'] = []
    net.variables['8s'] = []

    with tf.name_scope('Conv0'):
        with tf.name_scope('_1'):
            conv0_1, conv0_1vars = helpers.conv2d(input = net.input,
                                        filters = 32,
                                        layername='Conv0_1'
                                        )
        with tf.name_scope('_2'):
            conv0_2, conv0_2vars = helpers.conv2d(input = conv0_1,
                                        filters = 64,
                                        layername='Conv0_1'
                                        )
            net.variables['32s'].extend(conv0_1vars)
            net.variables['32s'].extend(conv0_2vars)
    with tf.name_scope('Conv0bis'):
        with tf.name_scope('_1'):
            conv0bis_1, conv0bis_1vars = helpers.conv2d_dilated(factor = 2,
                                                                input = conv0_2,
                                                                filters = 128,
                                                                layername='Conv0bis_1'
                                                                )
            net.variables['32s'].extend(conv0bis_1vars)
    with tf.name_scope('Pool0bis'):
        pool0bis = tf.layers.max_pooling2d(inputs=conv0bis_1,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool0bis = tf.layers.dropout(pool0bis, rate = 0.5)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,conv1_1vars = helpers.conv2d_dilated(input = pool0bis,
                                                         factor = 2,
                                                         filters = 256,
                                                         layername = 'Conv1_1'
                                                         )
            helpers.variable_summaries(conv1_1vars[0])
        with tf.name_scope('_2'):
            conv1_2,conv1_2vars = helpers.conv2d_dilated(input = conv1_1,
                                                         factor = 2,
                                                        filters = 256,
                                                            layername = 'Conv1_1',
                                                             ksize = [1,1]
                                                             )
            net.variables['32s'].extend(conv1_1vars)
            net.variables['32s'].extend(conv1_2vars)
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool1 = tf.layers.dropout(pool1,rate=0.2)
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d_dilated(input = pool1,
                                                          factor = 2,
                                                  filters = 256,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2, conv2_2vars = helpers.conv2d_dilated(input = conv2_1,
                                                          factor = 2,
                                       filters = 256,
                                       layername = 'Conv2_2',
                                       ksize = [1,1]
                                       )
            net.variables['32s'].extend(conv2_1vars)
            net.variables['32s'].extend(conv2_2vars)
    with tf.name_scope('Conv3'):
        conv3, conv3vars = helpers.conv2d_dilated(input=conv2_2,
                                                  filters = 256,
                                                  layername = 'Conv3',
                                                  factor = 4)
        net.variables['32s'].extend(conv3vars)
    with tf.name_scope('Conv4'):
        conv4, conv4vars = helpers.conv2d_dilated(input=conv3,
                                                  filters = 512,
                                                  layername = 'Conv4',
                                                  factor = 8
                                                  )
        net.variables['32s'].extend(conv4vars)
    with tf.name_scope('Merge'):
        pool1_resize, resize1vars = helpers.conv2d(input = pool1,
                                                   filters = net.numlabs,
                                                   layername = 'resizing1',
                                                   ksize = [3,3],
                                                   #k_init = [0,0]
                                                   )
        conv3_resize, resize2vars = helpers.conv2d(input = conv3,
                                                   filters = net.numlabs,
                                                   layername = 'resizing2',
                                                   ksize = [3,3],
                                                   #k_init = [0,0]
                                                   )
        conv4_resize, resize3vars = helpers.conv2d(input = conv4,
                                                   filters = net.numlabs,
                                                   layername = 'resizing3')
        helpers.variable_summaries(resize1vars[0])
        helpers.variable_summaries(resize2vars[0])
        helpers.variable_summaries(resize3vars[0])
        with tf.name_scope('pool1_resize'):
            helpers.variable_summaries(pool1_resize)
            helpers.inspect_layer(pool1_resize,3,'pool1')
        with tf.name_scope('conv3_resize'):
            helpers.variable_summaries(conv3_resize)
            helpers.inspect_layer(conv3_resize,3,'conv3')
        with tf.name_scope('conv4_resize'):
            helpers.variable_summaries(conv4_resize)
            helpers.inspect_layer(conv4_resize,3,'conv4')
        net.variables['8s'].extend(resize1vars)
        net.variables['16s'].extend(resize2vars)
        net.variables['32s'].extend(resize3vars)
        merged = tf.add(tf.add(pool1_resize,conv3_resize),conv4_resize)
    with tf.name_scope('Unpooling_2'):
        unpool2,unpool2vars = helpers.conv2d_transpose(merged,
                                                       net.numlabs,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = True
                                                       )
        net.variables['32s'].extend(unpool2vars)     
    
    with tf.name_scope('Unpooling_1'):
        unpool1, unpool1vars = helpers.conv2d_transpose(unpool2,
                                                        net.numlabs,
                                                        ksize=[3,3],
                                                        layername = 'unpool4',
                                                        relu = False
                                                        )
        net.variables['32s'].extend(unpool1vars)
    net.last_layer = unpool1
    net.compute_output()
    return net

def build_smallerCNN_upscaled_only2pooldeeper(input,numlab):
    net = Network(input,numlab=numlab)
    net.variables['32s'] = []
    net.variables['16s'] = []
    net.variables['8s'] = []

    with tf.name_scope('Conv0'):
        with tf.name_scope('_1'):
            conv0_1, conv0_1vars = helpers.conv2d(input = net.input,
                                        filters = 32,
                                        layername='Conv0_1'
                                        )
        with tf.name_scope('_2'):
            conv0_2, conv0_2vars = helpers.conv2d(input = conv0_1,
                                        filters = 64,
                                        layername='Conv0_1'
                                        )
            net.variables['32s'].extend(conv0_1vars)
            net.variables['32s'].extend(conv0_2vars)
    with tf.name_scope('Conv0bis'):
        with tf.name_scope('_1'):
            conv0bis_1, conv0bis_1vars = helpers.conv2d_dilated(factor = 2,
                                                                input = conv0_2,
                                                                filters = 128,
                                                                layername='Conv0bis_1'
                                                                )
            net.variables['32s'].extend(conv0bis_1vars)
    with tf.name_scope('Pool0bis'):
        pool0bis = tf.layers.max_pooling2d(inputs=conv0bis_1,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool0bis = tf.layers.dropout(pool0bis, rate = 0.5)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,conv1_1vars = helpers.conv2d_dilated(input = pool0bis,
                                                         factor = 2,
                                                         filters = 256,
                                                         layername = 'Conv1_1'
                                                         )
            helpers.variable_summaries(conv1_1vars[0])
        with tf.name_scope('_2'):
            conv1_2,conv1_2vars = helpers.conv2d_dilated(input = conv1_1,
                                                         factor = 2,
                                                        filters = 256,
                                                            layername = 'Conv1_1',
                                                             ksize = [1,1]
                                                             )
            net.variables['32s'].extend(conv1_1vars)
            net.variables['32s'].extend(conv1_2vars)
    with tf.name_scope('Pool1'):
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2,
                                        pool_size = [2,2],
                                        strides = 2,
                                        padding = 'SAME'
                                        )
        pool1 = tf.layers.dropout(pool1,rate=0.2)
        helpers.inspect_layer(pool1,depth=0,name='pool1')
        helpers.inspect_layer(pool1,depth=10,name='pool1')
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1, conv2_1vars = helpers.conv2d_dilated(input = pool1,
                                                          factor = 2,
                                                  filters = 256,
                                                  layername = 'Conv2_1'
                                                  )
        with tf.name_scope('_2'):
            conv2_2, conv2_2vars = helpers.conv2d_dilated(input = conv2_1,
                                                          factor = 2,
                                       filters = 256,
                                       layername = 'Conv2_2',
                                       ksize = [1,1]
                                       )
            net.variables['32s'].extend(conv2_1vars)
            net.variables['32s'].extend(conv2_2vars)
    with tf.name_scope('Conv3'):
        conv3, conv3vars = helpers.conv2d_dilated(input=conv2_2,
                                                  filters = 256,
                                                  layername = 'Conv3',
                                                  factor = 4)
        net.variables['32s'].extend(conv3vars)
    with tf.name_scope('Conv4'):
        conv4, conv4vars = helpers.conv2d_dilated(input=conv3,
                                                  filters = 512,
                                                  layername = 'Conv4',
                                                  factor = 8
                                                  )
        net.variables['32s'].extend(conv4vars)
    with tf.name_scope('Merge'):
        pool1_resize, resize1vars = helpers.conv2d(input = pool1,
                                                   filters = net.numlabs*8,
                                                   layername = 'resizing1',
                                                   ksize = [3,3],
                                                   #k_init = [0,0]
                                                   )
        conv3_resize, resize2vars = helpers.conv2d(input = conv3,
                                                   filters = net.numlabs*8,
                                                   layername = 'resizing2',
                                                   ksize = [3,3],
                                                   #k_init = [0,0]
                                                   )
        conv4_resize, resize3vars = helpers.conv2d(input = conv4,
                                                   filters = net.numlabs*8,
                                                   layername = 'resizing3')
        helpers.variable_summaries(resize1vars[0])
        helpers.variable_summaries(resize2vars[0])
        helpers.variable_summaries(resize3vars[0])
        with tf.name_scope('pool1_resize'):
            helpers.variable_summaries(pool1_resize)
            helpers.inspect_layer(pool1_resize,3,'pool1')
        with tf.name_scope('conv3_resize'):
            helpers.variable_summaries(conv3_resize)
            helpers.inspect_layer(conv3_resize,3,'conv3')
        with tf.name_scope('conv4_resize'):
            helpers.variable_summaries(conv4_resize)
            helpers.inspect_layer(conv4_resize,3,'conv4')
        net.variables['8s'].extend(resize1vars)
        net.variables['16s'].extend(resize2vars)
        net.variables['32s'].extend(resize3vars)
        merged = tf.add(tf.add(pool1_resize,conv3_resize),conv4_resize)
    with tf.name_scope('Unpooling_2'):
        unpool2,unpool2vars = helpers.conv2d_transpose(merged,
                                                       net.numlabs*4,
                                                       ksize=[3,3],
                                                       layername='unpool2',
                                                       relu = True
                                                       )
        net.variables['32s'].extend(unpool2vars)     
    
    with tf.name_scope('Unpooling_1'):
        unpool1, unpool1vars = helpers.conv2d_transpose(unpool2,
                                                        net.numlabs,
                                                        ksize=[3,3],
                                                        layername = 'unpool4',
                                                        relu = False
                                                        )
        net.variables['32s'].extend(unpool1vars)
    net.last_layer = unpool1
    net.compute_output()
    return net


def build_upscaled_nopool_noskip(input,numlab):

    net = Network(input,'net',numlab)
    net.variables['32s'] = []

    with tf.name_scope('Conv0'):
        with tf.name_scope('_1'):
            conv0_1, conv0_1vars = helpers.conv2d(input = net.input,
                                        filters = 16,
                                        layername='Conv0_1'
                                        )
        with tf.name_scope('_2'):
            conv0_2, conv0_2vars = helpers.conv2d(input = conv0_1,
                                        filters = 32,
                                        layername='Conv0_1'
                                        )
        net.variables['32s'].extend(conv0_1vars)
        net.variables['32s'].extend(conv0_2vars)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,conv1_1vars = helpers.conv2d_dilated(input = conv0_2,
                                                         filters = 64,
                                                         layername = 'Conv1_1',
                                                         factor = 2
                                                         )
        with tf.name_scope('_2'):
            conv1_2, conv1_2vars = helpers.conv2d_dilated(input = conv1_1,
                                                          filters = 64,
                                                          layername = 'Conv1_2',
                                                          factor = 2
                                                          )
        net.variables['32s'].extend(conv1_1vars)
        net.variables['32s'].extend(conv1_2vars)
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1,conv2_1vars = helpers.conv2d_dilated(input = conv1_2,
                                                         filters = 128,
                                                         layername = 'Conv2_1',
                                                         factor = 4
                                                         )
        with tf.name_scope('_2'):
            conv2_2, conv2_2vars = helpers.conv2d_dilated(input = conv2_1,
                                                          filters = 128,
                                                          layername = 'Conv2_2',
                                                          factor = 4
                                                          )
        net.variables['32s'].extend(conv2_1vars)
        net.variables['32s'].extend(conv2_2vars)
    with tf.name_scope('Conv3'):
        with tf.name_scope('_1'):
                conv3_1,conv3_1vars = helpers.conv2d_dilated(input = conv2_2,
                                                             filters = 128,
                                                             layername = 'Conv3_1',
                                                             factor = 8
                                                             )
        with tf.name_scope('_2'):
            conv3_2, conv3_2vars = helpers.conv2d_dilated(input = conv3_1,
                                                            filters = 128,
                                                            layername = 'Conv3_2',
                                                            factor = 8
                                                            )
        net.variables['32s'].extend(conv3_1vars)
        net.variables['32s'].extend(conv3_2vars)
    with tf.name_scope('Conv_4'):
        with tf.name_scope('_1'):
            conv4_1,conv4_1vars = helpers.conv2d_dilated(input = conv3_2,
                                                         filters = 256,
                                                         layername = 'Conv4_1',
                                                         factor = 16
                                                         )
        with tf.name_scope('_2'):
            conv4_2, conv4_2vars = helpers.conv2d(input = conv4_1,
                                                  filters = net.numlabs,
                                                  layername = 'Conv4_2',
                                                  ksize = [1,1],
                                                  relu = False
                                                  )
        net.variables['32s'].extend(conv4_1vars)
        net.variables['32s'].extend(conv4_2vars)

    net.last_layer = conv4_2
    net.compute_output()

    return net

def build_upscaled_nopool_noskip_deeper(input,numlab):

    net = Network(input,'net',numlab)
    net.variables['32s'] = []

    with tf.name_scope('Conv0'):
        with tf.name_scope('_1'):
            conv0_1, conv0_1vars = helpers.conv2d(input = net.input,
                                        filters = 32,
                                        layername='Conv0_1'
                                        )
        with tf.name_scope('_2'):
            conv0_2, conv0_2vars = helpers.conv2d(input = conv0_1,
                                        filters = 64,
                                        layername='Conv0_1'
                                        )
        net.variables['32s'].extend(conv0_1vars)
        net.variables['32s'].extend(conv0_2vars)
    with tf.name_scope('Conv1'):
        with tf.name_scope('_1'):
            conv1_1,conv1_1vars = helpers.conv2d_dilated(input = conv0_2,
                                                         filters = 128,
                                                         layername = 'Conv1_1',
                                                         factor = 2
                                                         )
        with tf.name_scope('_2'):
            conv1_2, conv1_2vars = helpers.conv2d_dilated(input = conv1_1,
                                                          filters = 128,
                                                          layername = 'Conv1_2',
                                                          factor = 2
                                                          )
        net.variables['32s'].extend(conv1_1vars)
        net.variables['32s'].extend(conv1_2vars)
    with tf.name_scope('Conv2'):
        with tf.name_scope('_1'):
            conv2_1,conv2_1vars = helpers.conv2d_dilated(input = conv1_2,
                                                         filters = 256,
                                                         layername = 'Conv2_1',
                                                         factor = 4
                                                         )
        with tf.name_scope('_2'):
            conv2_2, conv2_2vars = helpers.conv2d_dilated(input = conv2_1,
                                                          filters = 256,
                                                          layername = 'Conv2_2',
                                                          factor = 4
                                                          )
        net.variables['32s'].extend(conv2_1vars)
        net.variables['32s'].extend(conv2_2vars)
    with tf.name_scope('Conv3'):
        with tf.name_scope('_1'):
                conv3_1,conv3_1vars = helpers.conv2d_dilated(input = conv2_2,
                                                             filters = 256,
                                                             layername = 'Conv3_1',
                                                             factor = 8
                                                             )
        with tf.name_scope('_2'):
            conv3_2, conv3_2vars = helpers.conv2d_dilated(input = conv3_1,
                                                            filters = 256,
                                                            layername = 'Conv3_2',
                                                            factor = 8
                                                            )
        net.variables['32s'].extend(conv3_1vars)
        net.variables['32s'].extend(conv3_2vars)
    with tf.name_scope('Conv_4'):
        with tf.name_scope('_1'):
            conv4_1,conv4_1vars = helpers.conv2d_dilated(input = conv3_2,
                                                         filters = 512,
                                                         layername = 'Conv4_1',
                                                         factor = 16
                                                         )
        with tf.name_scope('_2'):
            conv4_2, conv4_2vars = helpers.conv2d(input = conv4_1,
                                                  filters = net.numlabs,
                                                  layername = 'Conv4_2',
                                                  ksize = [5,5],
                                                  relu = False
                                                  )
        net.variables['32s'].extend(conv4_1vars)
        net.variables['32s'].extend(conv4_2vars)

    net.last_layer = conv4_2
    net.compute_output()

    return net

def build_enet_small_nopool(input, numlab):

    net = Network(input, name = 'net', numlab = numlab)
    net.variables['all'] = []
    
    with tf.name_scope('Preprocess'):
        pre,prevar = helpers.conv2d_dilated(input = input,
                                            layername = 'preproc',
                                            filters = 8,
                                            factor = 2,
                                            bias = False
                                            )
        net.variables['all'].extend(prevar)

    with tf.name_scope('Stage_1'):

        with tf.name_scope('BNeck_1'):
            bneck11,bneck11vars = helpers.bottleneck(input = pre,
                                                    filters = 32,
                                                    scale = 2
                                                    )
            net.variables['all'].extend(bneck11vars)
        with tf.name_scope('BNeck_2'):
            bneck12,bneck12vars = helpers.bottleneck(input = bneck11,
                                                    filters = 64,
                                                    scale = 2
                                                    )
            net.variables['all'].extend(bneck12vars)
        with tf.name_scope('BNeck_3'):
            bneck13,bneck13vars = helpers.bottleneck(input = bneck12,
                                                    filters = 64,
                                                    scale = 2
                                                    )
            net.variables['all'].extend(bneck13vars)
        with tf.name_scope('BNeck_4'):
            bneck14,bneck14vars = helpers.bottleneck(input = bneck13,
                                                    filters = 64,
                                                    scale = 2
                                                    )
            net.variables['all'].extend(bneck14vars)

    with tf.name_scope('Stage_2'):
            
        with tf.name_scope('BNeck0'):
            bneck20,bneck20vars = helpers.bottleneck(input = bneck14,
                                                        filters = 128,
                                                        scale = 4
                                                        )
            net.variables['all'].extend(bneck20vars)
        with tf.name_scope('BNeck1'):
            bneck21,bneck21vars = helpers.bottleneck(input = bneck20,
                                                        filters = 128,
                                                        scale = 4
                                                        )
            net.variables['all'].extend(bneck21vars)
        with tf.name_scope('BNeck2'):
            bneck22,bneck22vars = helpers.bottleneck(input = bneck21,
                                                        filters = 128,
                                                        scale = 8
                                                        )
            net.variables['all'].extend(bneck22vars)
        with tf.name_scope('BNeck3'):
            bneck23,bneck23vars = helpers.bottleneck(input = bneck22,
                                                        filters = 128,
                                                        scale = 4,
                                                        asym = True
                                                        )
            net.variables['all'].extend(bneck23vars)
        with tf.name_scope('BNeck4'):
            bneck24,bneck24vars = helpers.bottleneck(input = bneck23,
                                                        filters = 128,
                                                        scale = 8
                                                        )
            net.variables['all'].extend(bneck24vars)
        with tf.name_scope('BNeck5'):
            bneck25,bneck25vars = helpers.bottleneck(input = bneck24,
                                                        filters = 128,
                                                        scale = 4
                                                        )
            net.variables['all'].extend(bneck25vars)
        with tf.name_scope('BNeck6'):
            bneck26,bneck26vars = helpers.bottleneck(input = bneck25,
                                                        filters = 128,
                                                        scale = 16
                                                        )
            net.variables['all'].extend(bneck26vars)
        with tf.name_scope('BNeck7'):
            bneck27,bneck27vars = helpers.bottleneck(input = bneck26,
                                                        filters = 128,
                                                        scale = 4,
                                                        asym = True
                                                        )
            net.variables['all'].extend(bneck27vars)
        with tf.name_scope('BNeck8'):
            bneck28,bneck28vars = helpers.bottleneck(input = bneck27,
                                                        filters = 128,
                                                        scale = 32
                                                        )
            net.variables['all'].extend(bneck28vars)
    with tf.name_scope('Decoder'):
        net.last_layer,decvars = helpers.conv2d(input = bneck28,
                                                filters = net.numlabs,
                                                layername = 'decoder',
                                                relu = False,
                                                bias = False
                                                )
        net.compute_output()
    return (net)

def build_context_agregg(input):

    aggreg = Network(input, name = 'aggreg', numlab = input.get_shape()[-1].value)
    aggreg.variables['all'] = []
    with tf.name_scope('Conv1'):
        aggreg.last_layer, vars1 = helpers.conv2d_dilated(input = input,
                                                   filters = aggreg.numlabs,
                                                   layername = 'conv1',
                                                   factor = 1
                                                   )
        aggreg.variables['all'].extend(vars1)
    with tf.name_scope('Conv2'):
        aggreg.last_layer, vars2 = helpers.conv2d_dilated(input = aggreg.last_layer,
                                                   filters = aggreg.numlabs,
                                                   layername = 'conv2',
                                                   factor = 1
                                                   )
        aggreg.variables['all'].extend(vars2)
    with tf.name_scope('Conv3'):
        aggreg.last_layer, vars3 = helpers.conv2d_dilated(input = aggreg.last_layer,
                                                   filters = aggreg.numlabs,
                                                   layername = 'conv3',
                                                   factor = 2
                                                   )
        aggreg.variables['all'].extend(vars3)
    with tf.name_scope('Conv4'):
        aggreg.last_layer, vars4 = helpers.conv2d_dilated(input = aggreg.last_layer,
                                                   filters = aggreg.numlabs,
                                                   layername = 'conv4',
                                                   factor = 4
                                                   )
        aggreg.variables['all'].extend(vars4)
    with tf.name_scope('Conv5'):
        aggreg.last_layer, vars5 = helpers.conv2d_dilated(input = aggreg.last_layer,
                                                   filters = aggreg.numlabs,
                                                   layername = 'conv5',
                                                   factor = 8
                                                   )
        aggreg.variables['all'].extend(vars5)
    with tf.name_scope('Conv6'):
        aggreg.last_layer, vars6 = helpers.conv2d_dilated(input = aggreg.last_layer,
                                                   filters = aggreg.numlabs,
                                                   layername = 'conv6',
                                                   factor = 16
                                                   )
        aggreg.variables['all'].extend(vars6)
    with tf.name_scope('Conv2'):
        aggreg.last_layer, vars7 = helpers.conv2d_dilated(input = aggreg.last_layer,
                                                   filters = aggreg.numlabs,
                                                   layername = 'conv7',
                                                   factor = 1
                                                   )
        aggreg.variables['all'].extend(vars7)
    with tf.name_scope('Conv8'):
        aggreg.last_layer, vars8 = helpers.conv2d_dilated(input = aggreg.last_layer,
                                                   filters = aggreg.numlabs,
                                                   layername = 'conv8',
                                                   factor = 1,
                                                   ksize = [1,1],
                                                   relu = False
                                                   )
        aggreg.variables['all'].extend(vars8)
    aggreg.compute_output()

    return aggreg