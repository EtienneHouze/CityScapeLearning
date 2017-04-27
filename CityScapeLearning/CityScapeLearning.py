"""
    This is the main script of the project.    
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import PIL
from os.path import join, normpath
from os import listdir
from Network import *
import helpers
import random



def perso_loss(logits, labs, weights):
    """
        Defines a weighted loss function.
        @ args :
            - logits = 4D tensor (batch*width*height*num_classes), the logits coded in the last dimension
            - labs : 3D tensor containing the labels (batch*width*height)
            - weights : a 2D tensor (batch*num_classes) tensors, containing the frequency of the different classes.
        @ returns :
            - a scalar, the mean across the batch of the weighted cross entropy
    """
    # range = tf.constant([0.5,num_classes-0.5],dtype=tf.float32)
    # hist = tf.histogram_fixed_width(values = tf.cast(labs,dtype=tf.float32),
    # nbins = tf.constant(num_classes,dtype = tf.int32),
    # value_range = range,
    # dtype = tf.float32)
    # with tf.name_scope('Weights'):
    #    helpers.variable_summaries(weights)
    epsilon = tf.constant(value=1e-30)
    labs_one = tf.one_hot(tf.cast(labs, dtype=tf.int32), weights.get_shape()[-1].value)
    weights_inv = 1/(weights+1e-10)
    weights_inv = tf.expand_dims(weights_inv,1)
    weights_inv = tf.expand_dims(weights_inv,1)
    weights_inv = tf.tile(weights_inv,multiples=[1,logits.get_shape()[1].value,logits.get_shape()[2].value,1])
    softmax = tf.nn.softmax(logits)
    #logits_flat = tf.reshape(softmax, shape=[-1, weights.get_shape()[-1].value])
    #label_flat = tf.reshape(labs, shape=[-1, weights.get_shape()[-1].value])
    return -tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.multiply(labs_one,
                                                                tf.log(softmax + epsilon)
                                                                ),
                                                     #weights_inv),
                                                     1),
                                         axis=[1,2,3]
                                         )
                           )


def loss(logits, labs):
    """
        Compute the loss cross entropy loss
        @ args :
            - logits : a 4D tensor containing the logits
            - label : a 3D tensor containing the labels
        @ returns :
            - a scalar, the mean of the cross entropy loss of the batch
    """
    flat_logits = tf.reshape(logits, shape=[-1, num_labs])
    flat_labels = tf.reshape(labs, shape=[-1])
    flat_labels_one_hot = tf.one_hot(tf.cast(flat_labels, dtype=tf.int32), num_labs)

    return (tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=flat_labels,
            logits=flat_logits
        ),
        name='Loss'
    )
    )


def total_loss(logits, label, weights, beta=0.0005, varslist = []):
    l = perso_loss(logits, label, weights)
    for var in varslist:
        l += beta * tf.nn.l2_loss(var)
    return l




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


def train(graphbuilder, batch_size=10, train_size=1000, epochs=3, train_dir='D:/EtienneData/trainsmalllesslabs', log_dir='/log',imW=256, imH=256, learning_rate=1e-4, num__labs=35, saving_path = None, loading_path = None, savestep = 500):
    """
       Defines an runs a training session.
       @ args :
            - batch_size : number of images per mini-batch
            - train_size : number of images from the training directory to use
            - epochs : number of epochs to perform
            - train_dir : path to the folder containing training images and labels
            - saver : path to the file to save the model
            - log_dir : path to the log folder for tensorboard info 
            - imW, imH : width and height of the images
            - learning rate : self explanatory
            - num_labs : either 35 or 8
            - saving_path : if defined, path to the checkpoints to save the model variables
            - loading_path : if defined, path to the model variables to load
            - savestep : how often the model will be saved.
    """
    num_labs = num__labs
    train_set, histo = helpers.produce_training_set(traindir=train_dir, trainsize=train_size,numlabs=num__labs)
    freqs = histo / np.sum(histo)
    weights = 1/freqs
    mainGraph = tf.Graph()
    with mainGraph.as_default():
        with tf.name_scope('Input'):
            ins = tf.placeholder(shape=(batch_size, imH, imW, 3),
                                    dtype=tf.float32)
            labs = tf.placeholder(shape=(batch_size, imH, imW),
                                    dtype=tf.int32)
            weigs = tf.placeholder(shape=(batch_size,num__labs),
                                    dtype=tf.float32)

        with tf.name_scope("Net"):
            CNN = graphbuilder(input=ins,numlab=num__labs)
        global_step = tf.Variable(initial_value=0,
                                    name='global_step',
                                    trainable=False)

        with tf.name_scope('out'):
            helpers.image_summaries(tf.expand_dims(input=CNN.output, axis=-1), name='output')
            helpers.variable_summaries(tf.cast(CNN.output, dtype=tf.float32))
        with tf.name_scope('labels'):
            helpers.image_summaries(tf.expand_dims(input=labs, axis=-1), name='labels')
            helpers.variable_summaries(tf.cast(labs, dtype=tf.float32))

        with tf.name_scope('Loss'):
            l = perso_loss(logits=CNN.last_layer, labs=labs, weights=weigs)
            tf.summary.scalar(name='loss', tensor=l)

        with tf.name_scope('Learning'):
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=l,                                                                global_step=global_step)


        merged = tf.summary.merge_all()
    
    with tf.Session(graph=mainGraph) as sess:

        trainWriter = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        if (loading_path):
            loader = tf.train.Saver()
            loader.restore(sess,loading_path)
        for epoch in range(epochs):
            random.shuffle(train_set)
            for i in range(int(train_size / batch_size)):
                if (i == 0):
                    [images, labels, w] = helpers.produce_mini_batch(train_set, step=i, imW=imW, imH=imH, batch_size=batch_size, numlabs=num__labs)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)
                    run_meta = tf.RunMetadata()
                    _, out, test_layer, test_loss, summary, step = sess.run(
                        (train_step, CNN.output, CNN.last_layer, l, merged, global_step),
                        feed_dict={ins: images, labs: labels, weigs : w})
                    print(test_loss, i, epoch)
                    trainWriter.add_run_metadata(run_meta, 'step%d' % step)
                    trainWriter.add_summary(summary, step)
                else:
                    [images, labels, w] = helpers.produce_mini_batch(train_set, step=i, imW=imW, imH=imH, batch_size=batch_size, numlabs = num__labs)
                    _, out, test_layer, test_loss, summary, step = sess.run(
                        (train_step, CNN.output, CNN.last_layer, l, merged, global_step),
                        feed_dict={ins: images, labs: labels, weigs : w})
                    print(test_loss, i, epoch)
                    trainWriter.add_summary(summary, step)
                if (saving_path and step % savestep == 0):
                    saver = tf.train.Saver()
                    print (saver.save(sess,
                               save_path = saving_path,
                               global_step = step,
                               write_meta_graph = False
                               )
                           )




#produce_training_dir(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',
                     #labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train', training_set_size=10000,
                     #imW=256, imH=128, outdir='D:/EtienneData/smalltrainresized', crop=False)

#test = helpers.produce_training_set_names(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',
#                                          labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',
#                                          trainsize=10000)
#batchsize = 10
#list = helpers.produce_batch_from_names(test,batchsize=batchsize,batch_number=0)
#gr = tf.Graph()
#with gr.as_default():
#    im_list = tf.placeholder(dtype=tf.string,shape=(batchsize))
#    lab_list = tf.placeholder(dtype=tf.string, shape=(batchsize))
#    ims,labs = helpers.produce_inputs(im_list,lab_list,batchsize,size=[128,256])
#with tf.Session(graph=gr) as sess:
#    sess.run(tf.global_variables_initializer())
#    listims, listlabs = sess.run((ims,labs),feed_dict = {im_list : list[0],lab_list : list[1]})
#    print('done')

def test(testdir, netbuilder, savedmodel, num__labs = 8, num_im = 100, batch_size = 1,imH = 128,imW = 256):
    
    
    #Building the model
    mainGraph = tf.Graph()
    with mainGraph.as_default():
        with tf.name_scope('Input'):
            ins = tf.placeholder(shape=(batch_size, imH, imW, 3),
                                    dtype=tf.float32)
            labs = tf.placeholder(shape=(batch_size, imH, imW),
                                    dtype=tf.int32)
            weigs = tf.placeholder(shape=(batch_size,num__labs),
                                    dtype=tf.float32)

        with tf.name_scope("Net"):
            CNN = netbuilder(input=ins,numlab=num__labs)
        #global_step = tf.Variable(initial_value=0,
        #                            name='global_step',
        #                            trainable=False)

        #with tf.name_scope('out'):
        #    helpers.image_summaries(tf.expand_dims(input=CNN.output, axis=-1), name='output')
        #    helpers.variable_summaries(tf.cast(CNN.output, dtype=tf.float32))
        #with tf.name_scope('labels'):
        #    helpers.image_summaries(tf.expand_dims(input=labs, axis=-1), name='labels')
        #    helpers.variable_summaries(tf.cast(labs, dtype=tf.float32))

        with tf.name_scope('Loss'):
            l = perso_loss(logits=CNN.last_layer, labs=labs, weights=weigs)
            #tf.summary.scalar(name='loss', tensor=l)

    
    with tf.Session(graph = mainGraph) as sess:
        sess.run(tf.global_variables_initializer())
        test_set = helpers.produce_testing_set(testdir, num_im,imH=imH,imW = imW)
        loader = tf.train.Saver()
        loader.restore(sess, save_path = savedmodel)
            
        for i in range(num_im):
                print(i)
                [images, labels, w] = helpers.produce_mini_batch(test_set, step=i, imW=imW, imH=imH, batch_size=batch_size, numlabs=num__labs)
                preds, test_loss = sess.run((CNN.output,l),feed_dict = {ins : images, labs : labels, weigs : w})
                print(test_loss)
                print('yeah')
                IOU = np.zeros((num__labs))
                for lab_ind in range(num__labs):
                    TP = 0.0
                    FP = 0.0
                    FN = 0.0
                    for j in range(imH):
                        for k in range(imW):
                            if (preds[0,j,k]==lab_ind and labels[0][j,k]==lab_ind):
                                TP += 1
                            elif (preds[0,j,k]==lab_ind and labels[0][j,k]!=lab_ind):
                                FP += 1
                            elif (preds[0,j,k]!=lab_ind and labels[0][j,k]==lab_ind):
                                FN +=1
                    IOU[lab_ind] = TP/(TP+FP+FN)
                IOU_mean = np.mean(IOU)
                print('mean IOU is : ' + str(IOU_mean))
