from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from PIL import Image
from os.path import join, basename, isfile, normpath
from os import listdir, walk
from labels import *


num_labels = 35

"""
    Preprocessing method to set up a training set from the cityscape folders.
    @ args :
        - imdir : string, the path to the training folder
        - labeldir : string, the path to the labels folder
        - training_set_size : an int, the number of image to use in the training set
        - imWW, imH : integers, the size of the cropped images we want to use in the training set.
    @ returns :
        - out : a list of tuples (imname,labelname,cornerX,cornerY), with :
            imname : the path to the image
            labelname : the path to the corresponding label
            conerX, cornerY : the coordinates of the top left corner in the image to crop.
"""
def produce_training_set(imdir,labeldir,training_set_size,imW=640,imH=360):
    filelist = []
    imdir = normpath(imdir)
    labeldir = normpath(labeldir)
    for path, subdirs, files in walk(imdir):
        for name in files:
            splt_name = str(basename(name)).split(sep="_")
            img_name = join(path,name)
            city = splt_name[0]
            label_name = join(normpath(labeldir),city,city+'_'+splt_name[1]+'_'+splt_name[2]+'_gtFine_labelIds.png')
            if (isfile(label_name)):
                filelist.append([img_name,label_name])
    out = []
    random_indices = np.random.randint(low=0,high=len(filelist),size=training_set_size)
    for i in random_indices:
        out.append(filelist[i]+[np.random.randint(2048-imW),np.random.randint(1024-imH)])
    return out

"""
    Produces a mini batch of images and corresponding labels.
    @ args :
        - trainingset : a list of tuples, see the output of the produce_training_set mathod
        - step : the step of the iteration inside the epoch
        - imW,imH = size of the cropped images
        - batch_size : length of a batch
    @ returns :
        - out : a list of pairs [im, label] with 
            - im : a ImW*ImH*3 float32 np array, encoding the cropped image
            - label : a ImW*ImH uint8 np array, encoding the labelled cropped image
"""
def produce_mini_batch(trainingset, step, imW=640, imH=360, batch_size = 10):
    batch_list = trainingset[batch_size*step:(step*batch_size)+batch_size]
    out_im = []
    out_lab = []
    for data in batch_list:
        Im = Image.open(data[0])
        Im = Im.crop((data[2],data[3],data[2]+imW,data[3]+imH))
        Label = Image.open(data[1])
        Label = Label.crop((data[2],data[3],data[2]+imW,data[3]+imH))
        im = np.asarray(Im,dtype=np.float32)
        label = np.asarray(Label,dtype=np.uint8)
        label_one_hot = np.eye(num_labels)[label]
        out_im.append(im)
        out_lab.append(label_one_hot)
    return [out_im, out_lab]


"""
    Shows the colored image according to the labels.
    @ args :
        - image : a imW*imH np array, containing labels for every pixel
    @ shows :
        - the colored image, with coloring done according to the labels define in the .py file.
"""
def show_labelled_image(image):
    out_view = np.zeros(shape=(image.shape[0],image.shape[1],3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lab = id2label[image[i,j]]
            out_view[i,j,:] = lab.color
    I = Image.fromarray(np.uint8(out_view))
    I.show()

"""
    Computes a max_pool layer with a scale factor of 2
    @ args :
        - input : a 4D-Tensor
    @ returns :
        - output : the 4D tensor output of the layer
        - pool_indices : the 4D Tensor memorizing the indices of the pooling operation
"""
def max_pool_with_mem(input):
    batch_size = input.get_shape()[0].value
    height = input.get_shape()[2].value
    width = input.get_shape()[1].value
    channels = input.get_shape()[3].value
    pool_indices = tf.zeros(shape=[batch_size,width/2,height/2,channels],dtype=tf.uint8)
    output = tf.Variable(tf.zeros(shape=pool_indices.get_shape(),dtype=input.dtype))
    for i in range(batch_size):
        for j in range(channels):
            for k in range(int(width/2)):
                for l in range(int(height/2)):
                    M = tf.reduce_max(input[i,2*k:2*k+1,2*l:2*l+1,j])
                    if (input[i,2*k,2*l,j] == M):
                        pool_indices[i,k,l,j] = 0
                    elif (input[i,2*k+1,2*l,j] == M):
                        pool_indices[i,k,l,j] = 1
                    elif (input[i,2*k,2*l+1,j] == M):
                        pool_indices[i,k,l,j] = 2
                    elif (input[i,2*k+1,2*l+1,j] == M):
                        pool_indices[i,k,l,j] = 3
                    output[i,k,l,j].value = M
    return (output, pool_indices)

def unpool(input,pool_indices):
    batch_size = input.get_shape()[0].value
    height = input.get_shape()[2].value
    width = input.get_shape()[1].value
    channels = input.get_shape()[3].value
    output = tf.zeros(shape=[batch_size,2*width,2*height,channels],dtype=input.dtype)
    for i in range(batch_size):
        for j in range(channels):
            for k in range(width):
                for l in range(height):
                    output[i,2*k+(pool_indices[i,k,l,j]%2),2*l+(pool_indices[i,k,l,j]/2),j] = input[i,k,l,j]
    return output
