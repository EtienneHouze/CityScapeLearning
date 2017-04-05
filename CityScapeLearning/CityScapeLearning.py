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

def weighted_loss(logits, lab, num_classes, weights):
    with tf.name_scope('loss_1'):

        logits_flat = tf.reshape(logits, [-1, num_classes])

        epsilon = tf.constant(value=1e-10)

        # consturct one-hot label array
        label_flat = tf.reshape(lab, [-1])

        # should be [batch ,num_classes]
        labels_hot = tf.one_hot(label_flat, depth=num_classes)

        softmax = tf.nn.softmax(logits_flat) 
        print()
        cross_entropy = -tf.reduce_sum(tf.multiply(tf.multiply(labels_hot,tf.log(softmax + epsilon)), weights),axis=1)
        #cross_entropy = -tf.reduce_sum(tf.multiply(labels_hot,tf.log(softmax + epsilon)),axis=1)
        #batch wise mean
        cross_entropy_mean=tf.reduce_mean(cross_entropy)

    return cross_entropy_mean


def perso_loss(logits,label):
    label = tf.one_hot(label,num_labels)
    epsilon = tf.constant(value=1e-10)
    logits_flat = tf.reshape(logits,shape=[-1,num_labels])
    softmax = tf.nn.softmax(logits_flat)
    label_flat = tf.reshape(label,shape=[-1,num_labels])
    return tf.reduce_mean(-tf.reduce_sum(tf.multiply(label_flat,tf.log(softmax + epsilon)),axis = 1))


def loss(logits,label):
    """
        Compute the loss cross entropy loss
        @ args :
            - logits : a 4D tensor containing the logits
            - label : a 3D tensor containing the labels
        @ returns :
            - a scalar, the mean of the cross entropy loss of the batch
    """
    flat_logits=tf.reshape(logits,shape=[-1,num_labels])
    flat_labels=tf.reshape(label,shape=[-1])

    return (tf.reduce_mean(
                          tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    labels=label,
                                                    logits=logits
                                                    ),
                          name = 'Loss'
                          )
            )

def build_CNN(graph,input):
    """
        Builds a fully convolutionnal neural network.
        @ args :
            - graph : the tf.Graph containing the net operations
            - input : the input tensor
        @ returns :
            - net : a Network object containing the net, as described in the paper by J.Long et al.
    """
    with graph.as_default():
        with tf.name_scope('CNN'):
                net = Network(input)
                net.add_complete_encoding_layer(depth=32,layerindex=0,pooling=True,bias=True,num_conv=3)
                net.add_complete_encoding_layer(depth=64,layerindex=1,pooling=True,bias=True,num_conv=3)
                net.add_complete_encoding_layer(depth=128,layerindex=2,num_conv=3,pooling=True,bias=True)
                net.add_complete_encoding_layer(depth=256,layerindex=3,num_conv=3,pooling=True,bias=True)
                net.add_complete_encoding_layer(depth=512,layerindex=4,num_conv=3,pooling=True,bias=True)
                net.add_complete_encoding_layer(depth=35,layerindex=5,num_conv=1,pooling=False,bias=False,relu = False,ksize=[8,16])
                net.add_complete_decoding_layer(corresponding_encoding=4,bias=False,num_conv=0)
                net.add_complete_decoding_layer(corresponding_encoding=3,bias=False,num_conv=0,init_weight=0.4)
                net.add_complete_decoding_layer(corresponding_encoding=2,bias=False,num_conv=0,init_weight=0.2)
                net.add_complete_decoding_layer(corresponding_encoding=1,bias=False,num_conv=0,init_weight=0)
                net.add_complete_decoding_layer(corresponding_encoding=0,bias=False,num_conv=0,init_weight=0)
                net.add_complete_encoding_layer(depth=35,layerindex=5,num_conv=1,pooling=False,bias=False,relu = False)
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

def train(batch_size = 10, train_size = 1000, epochs = 10, train_dir = 'D:/EtienneData/train', saver = None, log_dir = '/log'):
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
    train_set = produce_training_set(traindir = train_dir,trainsize = train_size)
    mainGraph = tf.Graph()
    
    with mainGraph.as_default():
        with tf.name_scope('Input'):
            ins = tf.placeholder(shape=(batch_size,256,512,3),
                                        dtype=tf.float32)
            labs = tf.placeholder(shape=(batch_size,256,512),
                                         dtype = tf.int32)
    CNN = build_CNN(mainGraph,input=ins)
    with mainGraph.as_default():
        global_step = tf.Variable(initial_value=0,
                                  name = 'global_step',
                                  trainable = False)
        #if (tf.equal(tf.mod(global_step,100),0)):
        #    show_labelled_image(CNN.output[0])
        #    show_labelled_image(labs[0])
        with tf.name_scope('Learning'):
            with tf.name_scope('Loss'):
                l = loss(logits=CNN.last_layer,label = labs)
            tf.summary.scalar(name='loss',tensor=l)
            train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss = l,
                                                                               global_step = global_step,
                                                                               var_list= CNN.encoder_variables+CNN.decoder_variables)
        merged = tf.summary.merge_all()


    with tf.Session(graph=mainGraph) as sess:
        trainWriter = tf.summary.FileWriter(logdir=log_dir,graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            random.shuffle(train_set)
            for i in range(int(train_size/batch_size)):
                if (i ==0):
                    [images, labels] = produce_mini_batch(train_set,step = i,imW=512,imH=256,batch_size=batch_size)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_meta = tf.RunMetadata()
                    _, out,test_layer,test_loss, summary = sess.run((train_step, CNN.output,CNN.last_layer,l,merged), feed_dict={ins : images, labs : labels})
                    print(test_loss,i,epoch)
                    trainWriter.add_run_metadata(run_meta,'step%d' % (int(epoch*trainsize/batch_size) + i))
                    trainWriter.add_summary(summary, int(epoch*trainsize/batch_size) + i)
                else :
                    [images, labels] = produce_mini_batch(train_set,step = i,imW=512,imH=256,batch_size=batch_size)
                    _, out,test_layer,test_loss, summary = sess.run((train_step, CNN.output,CNN.last_layer,l,merged), feed_dict={ins : images, labs : labels})
                    print(test_loss,i,epoch)
                    trainWriter.add_summary(summary, int(epoch*trainsize/batch_size) + i)

    