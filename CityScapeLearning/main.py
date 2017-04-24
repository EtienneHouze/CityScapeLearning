import CityScapeLearning as city
import helpers
from Network import *
from Model import Model


model_test = Model(name='model_2_ter',saving_dir='D:/EtienneData/models/day_13/4', netbuilder = build_smallerCNN_upscaled,imH=256,imW=512,num_labs=8,trained_vars=[])

model_test.train(train_dir = 'D:/EtienneData/trainmediumlesslabs', log_dir='log_day13/6', batch_size = 10, train_size = 10000, epochs = 10, learning_rate = 1e-4, trainable_vars = ['8s','16s','32s'],savestep=500)

model_test.test(testdir = 'D:/EtienneData/valless',num_im=10)

#model_test.compute('D:/EtienneData/valall/_4_im_.png')