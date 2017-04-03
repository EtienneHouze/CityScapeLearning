"""
    This is the main script of the project.    
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import PIL
from os.path import join
from os import listdir
from Network import *
from helpers import *

batch_size = 1

#set = produce_training_set(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',training_set_size=1000)
#batch = produce_mini_batch(trainingset=set, step = 0)
#im = PIL.Image.fromarray(np.uint8(batch[0][0]))
#lab = PIL.Image.fromarray(np.uint8(batch[0][1]))
#im.show()
##lab.show()
#show_labelled_image(batch[0][1])


def loss(logits,label):
    #num_labels = label._shape[-1].value
    #logits=tf.reshape(output,[logits._shape[0].value,-1,num_labels])
    #label = tf.reshape(label,[,-1])
    return (tf.reduce_sum(
                          tf.nn.softmax_cross_entropy_with_logits(
                                                    labels=label,
                                                    logits=logits,
                                                    dim=-1
                                                    ),
                          axis=[1,2],
                          name = 'Loss'
                          )
            )



mainGraph = tf.Graph()

#TODO : revoir comment est fait le reseau pour éviterr le OOM
#with mainGraph.as_default():
#    test_input = tf.Variable(initial_value=tf.random_normal(shape=[batch_size,640,360,3]),dtype=tf.float32)
#    test = Network(test_input)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=64)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=64)
#    test.add_MaxPool_Layer(factor=2)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=128)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=128)
#    test.add_MaxPool_Layer(factor=2)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=256)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=256)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=256)
#    test.add_MaxPool_Layer(factor=2)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=512)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=512)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=512)
#    test.add_MaxPool_Layer(factor=2)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=512)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=512)
#    test.add_conv_Layer(kernel_size=[3,3],padding="SAME",stride=[1,1,1,1],out_depth=512)
#    test.add_MaxPool_Layer(2)
#    test.add_conv_Layer(kernel_size=[20,12],padding="SAME",stride=[1,1,1,1],out_depth=2048)
#    test.add_conv_Layer(kernel_size=[1,1],padding="SAME",stride=[1,1,1,1],out_depth=2048)
#    test.add_conv_Layer(kernel_size=[1,1],padding="SAME",stride=[1,1,1,1],out_depth=1000,relu=False)
#    test.compute_output()
#    un, ind = max_pool_with_mem(test.output)

#with tf.device("/gpu:0"):
#    with tf.Session(graph=mainGraph) as sess:
#        sess.run(tf.global_variables_initializer())
#        out = sess.run(un)
#        print(out.shape)

with mainGraph.as_default():
    with tf.name_scope('Net'):
        test_input = tf.Variable(initial_value=tf.random_normal(shape=(batch_size,640,360,3)),dtype=tf.float32)
        test = Network(test_input)
        test.add_complete_encoding_layer(64,0,pooling=True)
        test.add_complete_encoding_layer(128,1,num_conv=3,pooling=True)
        test.add_complete_encoding_layer(256,2,num_conv=3,pooling=True)
        test.compute_output()
    merged = tf.summary.merge_all()

with tf.Session(graph=mainGraph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    out = sess.run(test.output)
    trainWriter = tf.summary.FileWriter('/log',sess.graph)
    print(out.shape)

