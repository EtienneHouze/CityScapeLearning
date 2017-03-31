from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from Network import *

class SegNet(Network):

    def __init__(self,input,num_stride=32,name='SegNet'):
        super(SegNet,self).__init__(name=name,input=input)
        #with tf.name_scope(self.name):



