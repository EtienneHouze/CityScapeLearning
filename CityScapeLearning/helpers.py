from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from PIL import Image
from os.path import join, basename, isfile, normpath
from os import listdir, walk

"""
    
"""
def produce_training_set(imdir,labeldir,training_set_size,imW=640,imH=360):
    filelist = []
    for path, subdirs, files in walk(imdir):
        for name in files:
            splt_name = str(basename(name)).split(sep="_")
            img_name = join(path,name)
            city = splt_name[0]
            label_name = join(normpath(labeldir),city,city+'_'+splt_name[1]+'_'+splt_name[2]+'_gtFine_labelIds.png')
            print(label_name)
            if (isfile(label_name)):
                filelist.append([img_name,label_name])
    out = []
    random_indices = np.random.randint(len(filelist),size=training_set_size)
    for i in random_indices:
        out.append(filelist[i]+[np.random.randint(2048-imW),np.random.randint(1024-imH)])

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
