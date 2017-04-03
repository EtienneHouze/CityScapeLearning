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

batch_size = 20

#set = produce_training_set(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',training_set_size=1000)
#batch = produce_mini_batch(trainingset=set, step = 0)
#im = PIL.Image.fromarray(np.uint8(batch[0][0]))
#lab = PIL.Image.fromarray(np.uint8(batch[0][1]))
#im.show()
##lab.show()
#show_labelled_image(batch[0][1])




def loss(logits,label):
    return (tf.reduce_sum(
                          tf.nn.softmax_cross_entropy_with_logits(
                                                    labels=label,
                                                    logits=logits,
                                                    dim=-1
                                                    ),
                          name = 'Loss'
                          )
            )



mainGraph = tf.Graph()

with mainGraph.as_default():
    with tf.name_scope('Input'):
        test_input = tf.placeholder(shape=(None,256,512,3),
                                    dtype=tf.float32)
        test_labels = tf.placeholder(shape=(None,256,512,35),
                                     dtype = tf.uint8)
    with tf.name_scope('Net'):
        test = Network(test_input)
        test.add_complete_encoding_layer(64,0,pooling=False,bias=False)
        #test.add_complete_encoding_layer(128,1,num_conv=3,pooling=True)
        #test.add_complete_encoding_layer(256,2,num_conv=3,pooling=True)
        test.add_complete_encoding_layer(35,1,num_conv=1,pooling=False,bias=False,relu = False)
        test.compute_output()
    l = loss(logits=test.last_layer,label = test_labels)
    tf.summary.scalar(name='loss',tensor=l)
    train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.).minimize(l,var_list=test.encoder_variables)
    merged = tf.summary.merge_all()

with tf.Session(graph=mainGraph) as sess:
    train_set = produce_training_set(imdir="D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train",labeldir="D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train",imW=512,imH=256,training_set_size=1000)
    sess.run(tf.global_variables_initializer())
    for i in range(int(1000/20)):
        [images, labels] = produce_mini_batch(train_set,step = 0,imW=512,imH=256,batch_size=20)
        _, out,test_loss = sess.run((train_step, test.output,l), feed_dict={test_input : images, test_labels : labels})
        print(test_loss)
    trainWriter = tf.summary.FileWriter('/log/5',sess.graph)
    print(out.shape)
    sess.close()
