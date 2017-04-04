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

batch_size = 5
trainsize = 500

#set = produce_training_set(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',training_set_size=1000)
#batch = produce_mini_batch(trainingset=set, step = 0)
#im = PIL.Image.fromarray(np.uint8(batch[0][0]))
#lab = PIL.Image.fromarray(np.uint8(batch[0][1]))
#im.show()
##lab.show()
#show_labelled_image(batch[0][1])




def loss(logits,label):
    label = tf.one_hot(label,num_labels)
    return (tf.reduce_mean(
                          tf.nn.softmax_cross_entropy_with_logits(
                                                    labels=label,
                                                    logits=logits,
                                                    dim=-1
                                                    ),
                          name = 'Loss'
                          )
            )



mainGraph = tf.Graph()
#produce_training_dir(imdir="D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train",labeldir="D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train",imW=512,imH=256,training_set_size=1000,outdir='D:/EtienneData/train')

with mainGraph.as_default():
    with tf.name_scope('Input'):
        test_input = tf.placeholder(shape=(batch_size,256,512,3),
                                    dtype=tf.float32)
        test_labels = tf.placeholder(shape=(batch_size,256,512),
                                     dtype = tf.uint8)
    with tf.name_scope('Net'):
        test = Network(test_input)
        test.add_complete_encoding_layer(depth=64,layerindex=0,pooling=True,bias=True)
        test.add_complete_encoding_layer(depth=128,layerindex=1,num_conv=3,pooling=True,bias=True)
        test.add_complete_encoding_layer(depth=256,layerindex=2,num_conv=3,pooling=True,bias=True)
        test.add_complete_decoding_layer(corresponding_encoding=2)
        test.add_complete_decoding_layer(corresponding_encoding=1)
        test.add_complete_decoding_layer(corresponding_encoding=0,bias=False,num_conv=3)
        test.add_complete_encoding_layer(35,4,num_conv=1,pooling=False,bias=False,relu = False)
        test.compute_output(top1=True)
    l = loss(logits=test.last_layer,label = test_labels)
    tf.summary.scalar(name='loss',tensor=l)
    train_step = tf.train.MomentumOptimizer(learning_rate=0.0001,momentum=0.9).minimize(l,var_list=test.encoder_variables+test.decoder_variables)
    merged = tf.summary.merge_all()

with tf.Session(graph=mainGraph) as sess:
    #train_set = produce_training_set(imdir="D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train",labeldir="D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train",imW=512,imH=256,training_set_size=1000)
    train_set = produce_training_set(traindir='D:/EtienneData/train',trainsize=500)
    sess.run(tf.global_variables_initializer())
    for i in range(int(500/batch_size)):
        [images, labels] = produce_mini_batch(train_set,step = i,imW=512,imH=256,batch_size=batch_size)
        _, out,test_layer,test_loss = sess.run((train_step, test.output,test.last_layer,l), feed_dict={test_input : images, test_labels : labels})
        print(test_loss)
        if (i%10 == 0):
            show_labelled_image(out[0])
    trainWriter = tf.summary.FileWriter('/log/6',sess.graph)
    print(out.shape)
    sess.close()
