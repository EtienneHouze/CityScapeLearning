from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from PIL import Image
from os.path import join, basename, isfile, normpath
from os import listdir, walk
import labels
import random

num_labels = len(labels.labels)
NUM_LABELS = len(labels.labels)

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

"""
    #def produce_training_set(imdir,labeldir,training_set_size,imW=640,imH=360):
    #    filelist = []
    #    imdir = normpath(imdir)
    #    labeldir = normpath(labeldir)
    #    for path, subdirs, files in walk(imdir):
    #        for name in files:
    #            splt_name = str(basename(name)).split(sep="_")
    #            img_name = join(path,name)
    #            city = splt_name[0]
    #            label_name = join(normpath(labeldir),city,city+'_'+splt_name[1]+'_'+splt_name[2]+'_gtFine_labelIds.png')
    #            if (isfile(label_name)):
    #                filelist.append([img_name,label_name])
    #    out = []
    #    random_indices = np.random.randint(low=0,high=len(filelist),size=training_set_size)
    #    step = 0
    #    for i in random_indices:
    #        Im = Image.open(filelist[i][0])
    #        x = np.random.randint(2048-imW)
    #        y = np.random.randint(1024-imH)
    #        Im = Im.crop((x,y,x+imW,y+imH))
    #        im = np.asarray(Im,dtype=np.float32)
    #        Label = Image.open(filelist[i][1])
    #        Label = Label.crop((x,y,x+imW,y+imH))
    #        label = np.asarray(Label,dtype=np.uint8)
    #        #label_one_hot = np.eye(num_labels)[label]
    #        out.append([im,label])
    #        step += 1
    #        print(step)
    #        #out.append(filelist[i]+[np.random.randint(2048-imW),np.random.randint(1024-imH)])
    #    return (out)
"""


def produce_training_dir(imdir, labeldir, outdir, training_set_size, imW=640, imH=360, crop=True, alllabels = True):
    """
        Creates a folder containing cropped images.
        @ args :
            - imdir : directory of the training images
            - labeldir : directory of the label images
            - outdir : path to the folder where we want to write the new cropped images
            - training_set_size : the number of images to write
            - imW, imH : width and height of the cropping to perform
            - crop : whether to crop or not the images
        @ returns :
            - nothing, simply writes images
    """
    filelist = []
    imdir = normpath(imdir)
    labeldir = normpath(labeldir)
    for path, subdirs, files in walk(imdir):
        for name in files:
            splt_name = str(basename(name)).split(sep="_")
            img_name = join(path, name)
            city = splt_name[0]
            label_name = join(normpath(labeldir), city,
                              city + '_' + splt_name[1] + '_' + splt_name[2] + '_gtFine_labelIds.png')
            if (isfile(label_name)):
                filelist.append([img_name, label_name])
    out = []
    random_indices = np.random.randint(low=0, high=len(filelist), size=training_set_size)
    step = 0
    for i in random_indices:
        if (alllabels):
            Im = Image.open(filelist[i][0])
            Label = Image.open(filelist[i][1])
            if (crop):
                x = np.random.randint(2048 - imW)
                y = np.random.randint(1024 - imH)
                Im = Im.crop((x, y, x + imW, y + imH))
                Label = Label.crop((x, y, x + imW, y + imH))
            else:
                Im.thumbnail((imW, imH))
                Label.thumbnail((imW, imW))
            Im.save(join(outdir, '_' + str(step) + '_im_.png'))
            Label.save(join(outdir, '_' + str(step) + '_lab_.png'))
        else:
            Im = Image.open(filelist[i][0])
            Label = Image.open(filelist[i][1])
            if (crop):
                x = np.random.randint(2048 - imW)
                y = np.random.randint(1024 - imH)
                Im = Im.crop((x, y, x + imW, y + imH))
                Label = Label.crop((x, y, x + imW, y + imH))
            else:
                Im.thumbnail((imW, imH))
                Label.thumbnail((imW, imW))
            Label = Image.eval(Label,labels.convert2catId)
            Im.save(join(outdir, '_' + str(step) + '_im_.png'))
            Label.save(join(outdir, '_' + str(step) + '_lab_.png'))
        print(step)
        step += 1
    return

def produce_training_set_names(imdir, labeldir, trainsize = None):
    """
        Produces a training set with the corresponding labels, as a list of paths.
        @ args :
            - traindir : path to the training directory
            - trainsize : number of images to produce in the training set. If none, the whole set is used.
        @ return :
            - out, a list of tuple (im_path, lab_path), with :
                -im_path : path to the image
                - lab_path : path to the corresponding label

    """
    filelist = []
    imdir = normpath(imdir)
    labeldir = normpath(labeldir)
    for path, subdirs, files in walk(imdir):
        for name in files:
            splt_name = str(basename(name)).split(sep="_")
            img_name = join(path,name)
            city = splt_name[0]
            label_name = join(normpath(labeldir), city,
                              city + '_' + splt_name[1] + '_' + splt_name[2] + '_gtFine_labelIds.png')
            if (isfile(label_name)):
                filelist.append([img_name, label_name])
    out = []
    if (trainsize):
        indices = np.random.randint(low = 0, high = len(filelist), size = (trainsize), dtype = np.int32)
        for i in indices:
            out.append(filelist[i])
    else :
        out = filelist
    return out

def produce_batch_from_names(namelist, batchsize, batch_number):
    """
        from a list of images and labels names, such as provided by the above function, returns a slice of those lists.
        @ args :
            - namelist : output of theproduce_training_set_names function
            - batchsize : length of  batch
            - batch_number : number of the batch to produce
        @ out :
            a tuple of lists of names of images and corresponding labels
    """
    
    im_names = []
    lab_names = []
    for i in range(batchsize*batch_number, batchsize*(batch_number+1)):
        im_names.append(namelist[i][0])
        lab_names.append(namelist[i][1])
    return (im_names, lab_names)

def produce_inputs(im_names, lab_names, batchsize, size=[1024,2048]):
    """
        From names of images and labels, reads and resize the images, output them as tensor.
        @ args :
            - im_names : a 1D tensor containing the names of the images to read.
            - lab_names : a 1D tensor containing the names of the labels to read.
            - size : the shape of the resized images.
        @ out :
            - ims = 
            - labs =
    """
    reader = tf.WholeFileReader()
    imqueue = tf.train.string_input_producer(im_names)
    labqueue = tf.train.string_input_producer(lab_names)
    #for i in range(batchsize):
    #    im_list.append(tf.expand_dims(tf.image.resize_images(tf.image.decode_png(im_names[i],
    #                                                                      channels=3,
    #                                                                      dtype=tf.uint8),
    #                                                  size=size),
    #                                  axis = 0)
    #                   )
    #    lab_list.append(tf.expand_dims(tf.image.resize_images(tf.image.decode_png(lab_names[i],
    #                                                                       channels=1,
    #                                                                       dtype = tf.uint8),
    #                                                   size=size),
    #                                   axis = 0)
    #                    )
    imkey, imval = reader.read(imqueue)
    labkey, labval = reader.read(labqueue)
    ims = tf.image.decode_png(imval,channels=3)
    labs = tf.image.decode_png(labval, channels=1)
    return ims,labs


def produce_training_set(traindir, trainsize,numlabs=35):
    """
        Produces a list of training images and labels.
        @ args :
            - traindir : path to the directory containing training images.
            - trainsize : an integer, the size of the training set we want to use. Must be lower than the number of images in the folde
        @ returns :
            - out : a list of pairs [im,lab], with 
                im : a 3D numpy array of the image
                lab : a 2D numpy array of the dense labels
    """

    num_labels = numlabs
    indices = list(range(trainsize))
    random.shuffle(indices)
    out = []
    hist = np.zeros((num_labels))
    for i in indices:
        Im = Image.open(normpath(join(traindir, '_' + str(i) + '_im_.png')))
        im = np.asarray(Im, dtype=np.float32)
        Label = Image.open(join(traindir, '_' + str(i) + '_lab_.png'))
        lab = np.asarray(Label.convert(mode="L"), dtype=np.float32)
##TODO
#        for j in range(lab.shape[0]):
#            for k in range(lab.shape[1]):
#                lab[j,k] = labels.id2catId[min((lab[j,k]-1,33))]
        new_hist, _ = np.histogram(lab, bins=num_labels)
        hist += new_hist
        out.append([im, lab])
    return out, hist



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


def produce_mini_batch(trainingset, numlabs,step, imW=640, imH=360, batch_size=10):
    batch_list = trainingset[batch_size * step:(step * batch_size) + batch_size]
    out_im = []
    out_lab = []
    weights = []
    for data in batch_list:
        out_im.append(data[0][:imH, :imW, :])
        out_lab.append(data[1][:imH, :imW])
        weights.append(np.histogram(a=out_lab[-1],bins=numlabs,density=True)[0])
    # Im = Image.open(data[0])
    #    Im = Im.crop((data[2],data[3],data[2]+imW,data[3]+imH))
    #    Label = Image.open(data[1])
    #    Label = Label.crop((data[2],data[3],data[2]+imW,data[3]+imH))
    #    im = np.asarray(Im,dtype=np.float32)
    #    label = np.asarray(Label,dtype=np.uint8)
    #    label_one_hot = np.eye(num_labels)[label]
    #    out_im.append(im)
    #    out_lab.append(label_one_hot)
    return [out_im, out_lab, weights]


def show_labelled_image(image, title=None):
    """
        Shows the colored image according to the labels.
        @ args :
            - image : a imW*imH np array, containing labels for every pixel
        @ shows :
            - the colored image, with coloring done according to the labels define in the .py file.
    """
    out_view = np.zeros(shape=(image.shape[0], image.shape[1], 3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j] == num_labels - 1):
                lab = id2label[-1]
            else:
                lab = id2label[image[i, j]]
            out_view[i, j, :] = lab.color
    with tf.name_scope(title):
        tf.summary.image(out_view)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def image_summaries(im, name='summary'):
    with tf.name_scope(name):
        tf.summary.image(tensor=tf.cast(x=im, dtype=tf.float32), name=name,max_outputs=1)

def inspect_layer(filter, depth, name='layer'):
    """
        Allows to summarize a layer at a desired depth.
        @args :
            - filter : ill-named layer, a 4D tensor
            - depth : the depth of the channel to scan
            - name : name of the layer to scan
        @ does :
            - adds an image summary of the desired layer
    """
    with tf.name_scope(name+str(depth)):
        tf.summary.image(tensor=tf.expand_dims(tf.cast(x=filter[:,:,:,depth], dtype = tf.float32), axis = -1), name=name+str(depth))

def convert_labelled_image(image):
    out_view = np.zeros(shape=(image.shape[0], image.shape[1], 3))
    for i in range(image.shape[1]):
        for j in range(image.shape[1]):
            lab = labels.id2label[image[i, j]]
            out_view[i, j, :] = lab.color
    return (out_view)





def max_pool_with_mem(input):
    """
        DEPRECATED
        Computes a max_pool layer with a scale factor of 2
        @ args :
            - input : a 4D-Tensor
        @ returns :
            - output : the 4D tensor output of the layer
            - pool_indices : the 4D Tensor memorizing the indices of the pooling operation
    """
    batch_size = input.get_shape()[0].value
    height = input.get_shape()[2].value
    width = input.get_shape()[1].value
    channels = input.get_shape()[3].value
    pool_indices = tf.zeros(shape=[batch_size, width / 2, height / 2, channels], dtype=tf.uint8)
    output = tf.Variable(tf.zeros(shape=pool_indices.get_shape(), dtype=input.dtype))
    for i in range(batch_size):
        for j in range(channels):
            for k in range(int(width / 2)):
                for l in range(int(height / 2)):
                    M = tf.reduce_max(input[i, 2 * k:2 * k + 1, 2 * l:2 * l + 1, j])
                    if (input[i, 2 * k, 2 * l, j] == M):
                        pool_indices[i, k, l, j] = 0
                    elif (input[i, 2 * k + 1, 2 * l, j] == M):
                        pool_indices[i, k, l, j] = 1
                    elif (input[i, 2 * k, 2 * l + 1, j] == M):
                        pool_indices[i, k, l, j] = 2
                    elif (input[i, 2 * k + 1, 2 * l + 1, j] == M):
                        pool_indices[i, k, l, j] = 3
                    output[i, k, l, j].value = M
    return (output, pool_indices)


def unpool(input, pool_indices):
    batch_size = input.get_shape()[0].value
    height = input.get_shape()[2].value
    width = input.get_shape()[1].value
    channels = input.get_shape()[3].value
    output = tf.zeros(shape=[batch_size, 2 * width, 2 * height, channels], dtype=input.dtype)
    for i in range(batch_size):
        for j in range(channels):
            for k in range(width):
                for l in range(height):
                    output[i, 2 * k + (pool_indices[i, k, l, j] % 2), 2 * l + (pool_indices[i, k, l, j] / 2), j] = \
                        input[i, k, l, j]
    return output

def conv2d(input, filters, layername, k_init=[0.0,0.1], stride = [1,1], padding = 'SAME',relu=True,bias=True, ksize = [3,3]):
    """
        Basically does the same thing as the built-in conv2D, but returns variables as well, to help monitor them...
        @args :
            - input : a 4D tensor, the input of the convolution
            - filters : an int, the number of channels of the output
            - layername : a string, the name of the operation
            - ksize : the size of the convolution kernel
            - k_init : a list of 2 float, the mean and the stddev used to initiate the convolution filter
            - stride : a list of 2 integers, the strides of the convolution
            - padding : explicit, must be 'SAME' or 'VALID'
            - relu : boolean, whether to relu the output or not
            - bias : a boolean, whether to use bias or not
        @out :
            - out : the 4D output tensor of the operation
            - vars : a list of the variables used int the convolution (length 1 or 2, depending of the use of bias)

    """
    with tf.name_scope(layername+'_Variables'):
        vars = []
        with tf.name_scope('Kernel'):
            K = tf.Variable(initial_value=tf.random_uniform(shape=[ksize[0],ksize[1],input.get_shape()[-1].value,filters],
                                                              minval = k_init[0] - k_init[1],
                                                              maxval = k_init[0] + k_init[1]),
                            dtype = tf.float32
                            )
            vars.append(K)
        if (bias):
            with tf.name_scope('Bias'):
                B = tf.Variable(initial_value = np.zeros(shape=[filters]),
                                dtype = tf.float32
                                )
                vars.append(B)
    if (bias):
        out = tf.add(tf.nn.conv2d(input = input,
                                filter = K,
                                strides = [1, stride[0], stride[1], 1],
                                padding = padding
                                ),
                     B)
    else:
        out = tf.nn.conv2d(input = input,
                                filter = K,
                                strides = [1, stride[0], stride[1], 1],
                                padding = padding
                                )
    if (relu):
        return (tf.nn.relu(out),vars)
    else:
        return (out,vars)


def conv2d_transpose(input, filters, layername, k_init=[0.0,0.1], stride = [2,2], padding = 'SAME',relu=True,bias=True, ksize = [3,3]):
    """
        Basically does the same thing as the built-in conv2D_transpose, but returns variables as well, to help monitor them...
        @args :
            - input : a 4D tensor, the input of the convolution
            - filters : an int, the number of channels of the output
            - layername : a string, the name of the operation
            - ksize : the size of the convolution kernel
            - k_init : a list of 2 float, the mean and the stddev used to initiate the convolution filter
            - stride : a list of 2 integers, the strides of the convolution
            - padding : explicit, must be 'SAME' or 'VALID'
            - relu : boolean, whether to relu the output or not
            - bias : a boolean, whether to use bias or not
        @out :
            - out : the 4D output tensor of the operation
            - vars : a list of the variables used int the convolution (length 1 or 2, depending of the use of bias)

    """
    with tf.name_scope(layername+'_Variables'):
        vars = []
        with tf.name_scope('Kernel'):
            K = tf.Variable(initial_value=tf.random_uniform(shape=[ksize[0],ksize[1],filters,input.get_shape()[-1].value],
                                                              minval = k_init[0] - k_init[1],
                                                              maxval = k_init[0] + k_init[1]),
                            dtype = tf.float32
                            )
            vars.append(K)
        if (bias):
            with tf.name_scope('Bias'):
                B = tf.Variable(initial_value = np.zeros(shape=[filters]),
                                dtype = tf.float32
                                )
                vars.append(B)
    if (bias) :
        out = tf.add(tf.nn.conv2d_transpose(value = input,
                                            filter = K,
                                             output_shape = [input.get_shape()[0].value,input.get_shape()[1].value * 2, input.get_shape()[2].value * 2, filters],
                                             strides = [1, stride[0], stride[1], 1],
                                            padding = padding
                                            ),
                     B)
    else :
        out = tf.nn.conv2d_transpose(value = input,
                                    filter = K,
                                    output_shape = [input.get_shape()[0].value,input.get_shape()[1].value * 2, input.get_shape()[2].value * 2, filters],
                                    strides = [1, stride[0], stride[1], 1],
                                    padding = padding
                                    )
    if (relu):
        return (tf.nn.relu(out),vars)
    else:
        return (out,vars)

#def merge_layers(largelayer, smalllayer, num_of_ups=1, filters = num_labels):
    #vars = []
    #with tf.name_scope("equalizing_depths"):
    #    small_layer,convvarS = conv2d(input=smalllayer,
    #                           filters = filters,
    #                           layername='',
    #                           ksize=[1,1],relu=False
    #                           )
    #    vars.append(convvarS)
    #    large_layer, convvarL = conv2d(input=largelayer,
    #                            filters = filters,
    #                            layername='',
    #                            ksize=[1,1],
    #                            relu = False
    #                            )
    #    vars.append(convvarL)
    #with tf.name_scope('unpooling_before_merge'):
    #    for i in range(num_of_ups):
    #        small_layer,unconvvar = conv2d_transpose(input = small_layer,
    #                                       layername = '_'+str(i),
    #                                       filters = largelayer.get_shape()[-1].value,
    #                                       relu = False
    #                                       )
    #        vars.append(unconvvar)
    #inspect_layer(small_layer,0,'small')
    #inspect_layer(largelayer, 0, 'large')
    #variable_summaries(small_layer)
    #variable_summaries(largelayer)
    ##return (tf.add(tf.nn.l2_normalize(largelayer,dim=-1),tf.nn.l2_normalize(small_layer,dim=-1),name='merging'),vars)
    #return (tf.add(largelayer,small_layer, name = 'mergin'), vars)

def predictions(input,inspect=True):
    out, _ = conv2d(input=input,
                    filters=num_labels,
                    layername = 'prediction',
                    relu = False,
                    ksize = [1,1]
                    )
    out = tf.nn.softmax(out,dim=-1)
    if (inspect):
        with tf.name_scope('inspecting'):
            preds = tf.argmax(tf.nn.softmax(out),
                                            axis=len(out.get_shape()[1:].as_list()),
                                            name='output')
            image_summaries(tf.expand_dims(input=preds, axis=-1), name='peek')

    return out

def alternate_merge(inputs,ksize=[1,1]):
    with tf.name_scope('Concatenation'):
        merged_layer = tf.concat(values = inputs,
                                 axis = -1
                                 )
    with tf.name_scope('Merging_conv'):
        return conv2d(merged_layer,
                      filters = num_labels,
                      layername = 'merging_conv',
                      bias = False,
                      ksize = ksize
                      )
