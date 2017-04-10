"""
    This is the main script of the project.    
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import PIL
from os.path import join
from os import listdir
from Network import *
from helpers import *

batch_size = 10
trainsize = 1000

#set = produce_training_set(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',training_set_size=1000)
#batch = produce_mini_batch(trainingset=set, step = 0)
#im = PIL.Image.fromarray(np.uint8(batch[0][0]))
#lab = PIL.Image.fromarray(np.uint8(batch[0][1]))
#im.show()
##lab.show()
#show_labelled_image(batch[0][1])


def perso_loss(logits,labs, weights):
    """
        Defines a weighted loss function.
        @ args :
            - logits = 4D tensor (batch*width*height*num_classes), the logits coded in the last dimension
            - labs : 3D tensor containing the labels (batch*width*height)
            - num_classes : number of classes
        @ returns :
            - a scalar, the mean across the batch of the weighted cross entropy
    """
    #range = tf.constant([0.5,num_classes-0.5],dtype=tf.float32)
    #hist = tf.histogram_fixed_width(values = tf.cast(labs,dtype=tf.float32),
                                    #nbins = tf.constant(num_classes,dtype = tf.int32),
                                    #value_range = range,
                                    #dtype = tf.float32)
    #with tf.name_scope('Weights'):
    #    helpers.variable_summaries(weights)
    epsilon = tf.constant(value=1e-10)
    num_classes = len(weights)
    labs = tf.one_hot(tf.cast(labs,dtype=tf.int32),num_classes)
    logits_flat = tf.reshape(logits,shape=[-1,num_classes])
    softmax = tf.nn.softmax(logits_flat)
    label_flat = tf.reshape(labs,shape=[-1,num_classes])
    return tf.reduce_mean(-tf.reduce_sum(tf.multiply(tf.multiply(label_flat,tf.log(softmax + epsilon)),weights),axis = 1))


def loss(logits,labs):
    """
        Compute the loss cross entropy loss
        @ args :
            - logits : a 4D tensor containing the logits
            - label : a 3D tensor containing the labels
        @ returns :
            - a scalar, the mean of the cross entropy loss of the batch
    """
    flat_logits=tf.reshape(logits,shape=[-1,num_labels])
    flat_labels=tf.reshape(labs,shape=[-1])

    return (tf.reduce_mean(
                          tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    labels=flat_labels,
                                                    logits=flat_logits
                                                    ),
                          name = 'Loss'
                          )
            )

def total_loss(logits,label,beta=0.0005):
    return loss(logits,label) + beta * tf.nn.l2_loss(logits)

def build_CNN(input):
    """
        Builds a fully convolutionnal neural network.
        @ args :
            - graph : the tf.Graph containing the net operations
            - input : the input tensor
        @ returns :
            - net : a Network object containing the net, as described in the paper by J.Long et al.
    """
    with tf.name_scope('CNN'):
            net = Network(input)
            net.add_complete_encoding_layer(depth=32,layerindex=0,pooling=True,bias=False,num_conv=1)
            net.add_complete_encoding_layer(depth=64,layerindex=1,pooling=True,bias=True,num_conv=2,monitor=False)
            net.add_complete_encoding_layer(depth=128,layerindex=2,num_conv=2,pooling=True,bias=True)
            net.add_complete_encoding_layer(depth=256,layerindex=3,num_conv=2,pooling=True,bias=True)
            net.add_complete_encoding_layer(depth=512,layerindex=4,num_conv=2,pooling=True,bias=True)
            #net.add_complete_encoding_layer(depth = 512, layerindex=5,num_conv = 2, pooling = True, bias = True)
            net.add_complete_encoding_layer(depth=256,layerindex=4,num_conv=1,pooling=False,bias=False,relu = True,ksize=[8,4])
            #net.add_complete_decoding_layer(corresponding_encoding=6,num_conv=0,bias = False)
            net.add_complete_decoding_layer(corresponding_encoding=4,bias=True,num_conv=1, init_weight=0.5)
            #net.add_complete_encoding_layer(depth=128,layerindex=10,num_conv=1,bias=False,pooling=False)
            #net.add_complete_decoding_layer(corresponding_encoding=4,bias=True,num_conv=1,init_weight=0.5)
            net.add_complete_decoding_layer(corresponding_encoding=3,bias=True,num_conv=1,init_weight=0.5)
            net.add_complete_decoding_layer(corresponding_encoding=2,bias=False,num_conv=1,init_weight=0.5)
            net.add_complete_decoding_layer(corresponding_encoding=1,bias=False,num_conv=1,init_weight=0.5)
            net.add_complete_decoding_layer(corresponding_encoding=0,bias=False,num_conv=1,init_weight=0.5)
            net.add_complete_encoding_layer(depth=35,layerindex=7,num_conv=1,pooling=False,bias=False,ksize=[5,5],relu = False)
            net.compute_output(top1=True)
    return net

def build_graph(input):
    net = Network(input)
    with tf.name_scope('Inputs'):
        input_shape = input.get_shape().as_list()

    with tf.name_scope('Conv_1 + Pool'):

        W_conv11 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,input_shape[-1],32], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv11 = tf.Variable(initial_value=0.1*tf.ones(shape=[32]),dtype=tf.float32)
        h_conv11 = tf.nn.relu(tf.nn.conv2d(input,W_conv11,strides=[1,1,1,1],padding="SAME")+b_conv11,name = "Conv1")

        W_conv12 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,32,64], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv12 = tf.Variable(initial_value=0.1*tf.ones(shape=[64]),dtype=tf.float32)
        h_conv12 = tf.nn.relu(tf.nn.conv2d(h_conv11,W_conv12,strides=[1,1,1,1],padding="SAME")+b_conv12, name = "Conv2")

        h_pool1 = tf.nn.max_pool(h_conv12,ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME", name="Pooling")

    with tf.name_scope("Conv_2 + Pool"):
        W_conv21 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,64,128], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv21 = tf.Variable(initial_value=0.1*tf.ones(shape=[128]),dtype=tf.float32)
        h_conv21 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv21,strides=[1,1,1,1],padding="SAME")+b_conv21, name = "Conv1")

        W_conv22 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,128,256], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv22 = tf.Variable(initial_value=0.1*tf.ones(shape=[256]),dtype=tf.float32)
        h_conv22 = tf.nn.relu(tf.nn.conv2d(h_conv21,W_conv22,strides=[1,1,1,1],padding="SAME")+b_conv22, name = "Conv2")

        h_pool2 = tf.nn.max_pool(h_conv22,ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = "Pooling")

    with tf.name_scope('Conv_3'):
        W_conv31 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,256,256], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv31 = tf.Variable(initial_value=0.1*tf.ones(shape=[256]),dtype=tf.float32)
        h_conv31 = tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv31,strides=[1,1,1,1],padding="SAME")+b_conv31, name = "Conv1")

        W_conv32 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,256,256], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv32 = tf.Variable(initial_value=0.1*tf.ones(shape=[256]),dtype=tf.float32)
        h_conv32 = tf.nn.relu(tf.nn.conv2d(h_conv31,W_conv32,strides=[1,1,1,1],padding="SAME")+b_conv32, name="Conv2")

    with tf.name_scope('Upscaling_1'):
        h_up4 = tf.image.resize_bilinear(h_conv32,size=[int(input_shape[1]/2),int(input_shape[2]/2)])
        h_merged4 = tf.concat([h_conv22,h_up4],axis=-1)
    with tf.name_scope('Conv_4'):
        W_conv41 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,512,256], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv41 = tf.Variable(initial_value=0.1*tf.ones(shape=[256]),dtype=tf.float32)
        h_conv41 = tf.nn.relu(tf.nn.conv2d(h_merged4,W_conv41,strides=[1,1,1,1],padding="SAME")+b_conv41, name = "Conv1")

        W_conv42 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,256,256], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv42 = tf.Variable(initial_value=0.1*tf.ones(shape=[256]),dtype=tf.float32)
        h_conv42 = tf.nn.relu(tf.nn.conv2d(h_conv41,W_conv42,strides=[1,1,1,1],padding="SAME")+b_conv42, name = "Conv2")

    with tf.name_scope('Upscaling_2'):
        h_up5 = tf.image.resize_bilinear(h_conv42,size=[int(input_shape[1]),int(input_shape[2])])
        h_merged5 = tf.concat([h_conv12,h_up5],axis=-1)

    with tf.name_scope('Conv_5'):
        W_conv51 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,256+64,256], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv51 = tf.Variable(initial_value=0.1*tf.ones(shape=[256]),dtype=tf.float32)
        h_conv51 = tf.nn.relu(tf.nn.conv2d(h_merged5,W_conv51,strides=[1,1,1,1],padding="SAME")+b_conv51, name = "Conv1")

        W_conv52 = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,256,num_labels], stddev = 0.1, dtype = tf.float32)
                               )
        b_conv52 = tf.Variable(initial_value=0.1*tf.ones(shape=[num_labels]),dtype=tf.float32)
        output = tf.add(tf.nn.conv2d(h_conv51,W_conv52,strides=[1,1,1,1],padding="SAME"),b_conv52, name = "Conv2")

    with tf.name_scope('Output'):
        net.last_layer = output
        net.compute_output(top1=True)

    return net
"""
    mainGraph = tf.Graph()
    produce_training_dir(imdir="D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train",labeldir="D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train",imW=512,imH=256,training_set_size=1000,outdir='D:/EtienneData/train')

    with mainGraph.as_default():
        with tf.name_scope('Input'):
            test_input = tf.placeholder(shape=(batch_size,256,512,3),
                                        dtype=tf.float32)
            test_labels = tf.placeholder(shape=(batch_size,256,512),
                                         dtype = tf.int32)
        with tf.name_scope('Net'):
            test = Network(test_input)
            test.add_complete_encoding_layer(depth=3,layerindex=0,pooling=True,bias=True,num_conv=0)
            test.add_complete_encoding_layer(depth=64,layerindex=1,pooling=True,bias=True)
            test.add_complete_encoding_layer(depth=128,layerindex=2,num_conv=3,pooling=True,bias=True)
            test.add_complete_encoding_layer(depth=256,layerindex=3,num_conv=3,pooling=True,bias=True)
            test.add_complete_encoding_layer(depth=512,layerindex=4,num_conv=3,pooling=True,bias=True)
            test.add_complete_encoding_layer(depth=35,layerindex=5,num_conv=1,pooling=False,bias=False,relu = False,ksize=[8,16])
            test.add_complete_decoding_layer(corresponding_encoding=4,bias=False,num_conv=0)
            test.add_complete_decoding_layer(corresponding_encoding=3,bias=False,num_conv=0,init_weight=0.4)
            test.add_complete_decoding_layer(corresponding_encoding=2,bias=False,num_conv=0,init_weight=0.2)
            test.add_complete_decoding_layer(corresponding_encoding=1,bias=False,num_conv=0,init_weight=0)
            test.add_complete_decoding_layer(corresponding_encoding=0,bias=False,num_conv=0,init_weight=0)
            test.add_complete_encoding_layer(depth=35,layerindex=5,num_conv=1,pooling=False,bias=False,relu = False)
            test.compute_output(top1=True)
        with tf.name_scope('Learning'):
            with tf.name_scope('Loss'):
                l = loss(logits=test.last_layer,label = test_labels)
            tf.summary.scalar(name='loss',tensor=l)
            train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(l,var_list=test.encoder_variables+test.decoder_variables)
        merged = tf.summary.merge_all()

    with tf.Session(graph=mainGraph) as sess:
        train_set = produce_training_set(traindir='D:/EtienneData/train',trainsize=trainsize)
        sess.run(tf.global_variables_initializer())
        trainWriter = tf.summary.FileWriter('/log/18',sess.graph)
        for epoch in range(100):
            random.shuffle(train_set)
            for i in range(int(trainsize/batch_size)):
                [images, labels] = produce_mini_batch(train_set,step = i,imW=512,imH=256,batch_size=batch_size)
                _, out,test_layer,test_loss, summary = sess.run((train_step, test.output,test.last_layer,l,merged), feed_dict={test_input : images, test_labels : labels})
                print(test_loss,i,epoch)
                trainWriter.add_summary(summary, int(epoch*trainsize/batch_size) + i)
                if (i%100 == 0):
                    show_labelled_image(out[0],title='output')
                    show_labelled_image(labels[0],title='label')

    
        print(out.shape)
        sess.close()
"""

def train(batch_size = 10, train_size = 1000, epochs = 10, train_dir = 'D:/EtienneData/train', saver = None, log_dir = '/log',imW=256,imH=256, learning_rate=1e-4):
    """
       Defines an runs a training session.
       @ args :
            - batch_size : number of images per mini-batch
            - train_size : number of images from the training directory to use
            - epochs : number of epochs to perform
            - train_dir : path to the folder containing training images and labels
            - saver : path to the file to save the model
            - log_dir : path to the log folder for tensorboard info 
    """
    train_set, histo = produce_training_set(traindir = train_dir,trainsize = train_size)
    freqs = histo / np.sum(histo)
    weights = np.power((1-freqs),3)
    mainGraph = tf.Graph()
    
    with mainGraph.as_default():
        with tf.name_scope('Input'):
            ins = tf.placeholder(shape=(batch_size,imH,imW,3),
                                        dtype=tf.float32)
            labs = tf.placeholder(shape=(batch_size,imH,imW),
                                         dtype = tf.int32)

        with tf.name_scope("Net"):
            #CNN = build_CNN(input=ins)
            CNN = build_graph(input=ins)
        
        global_step = tf.Variable(initial_value=0,
                                      name = 'global_step',
                                      trainable = False)

        with tf.name_scope('out'):    
            image_summaries(tf.expand_dims(input = CNN.output, axis = -1),name='output')
            variable_summaries(tf.cast(CNN.output,dtype = tf.float32))
        with tf.name_scope('labels'):
            image_summaries(tf.expand_dims(input = labs, axis = -1),name='labels') 
            variable_summaries(tf.cast(labs,dtype=tf.float32))

        with tf.name_scope('Loss'):
            l = loss(logits=CNN.last_layer,labs = labs)
            tf.summary.scalar(name='loss',tensor=l)
            
        with tf.name_scope('Learning'):
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss = l,
                                                                                global_step = global_step)
        merged = tf.summary.merge_all()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(graph=mainGraph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        trainWriter = tf.summary.FileWriter(logdir=log_dir,graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            random.shuffle(train_set)
            for i in range(int(train_size/batch_size)):
                if (i ==0):
                    [images, labels] = produce_mini_batch(train_set,step = i,imW=imW,imH=imH,batch_size=batch_size)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)
                    run_meta = tf.RunMetadata()
                    _, out,test_layer,test_loss, summary, step = sess.run((train_step, CNN.output,CNN.last_layer,l,merged, global_step), feed_dict={ins : images, labs : labels})
                    print(test_loss,i,epoch)
                    trainWriter.add_run_metadata(run_meta,'step%d' % step)
                    trainWriter.add_summary(summary, step)
                else :
                    [images, labels] = produce_mini_batch(train_set,step = i,imW=imW,imH=imH,batch_size=batch_size)
                    _, out,test_layer,test_loss, summary, step = sess.run((train_step, CNN.output,CNN.last_layer,l,merged, global_step), feed_dict={ins : images, labs : labels})
                    print(test_loss,i,epoch)
                    trainWriter.add_summary(summary, step)


#train(train_dir = 'D:/EtienneData/smalltrain',log_dir='log_day4/13',batch_size=5,epochs=10,train_size=1000,learning_rate=1e-4)
   
produce_training_dir(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',training_set_size=10000,imW=256,imH=128,outdir='D:/EtienneData/smalltrainresized',crop=False)