import CityScapeLearning as city
import helpers
from Network import *
from Model import Model


model_test = Model(name='model_4',saving_dir='C:/Users/Etienne.Houze/OneDrive - Bentley Systems, Inc/EtienneData/models/day_16', netbuilder = build_upscaled_nopool_noskip,imH=200,imW=512,num_labs=8,trained_vars=['32s'], last_cp = 'C:/Users/Etienne.Houze/OneDrive - Bentley Systems, Inc/EtienneData/models/day_16/model_4-86000')

#model_test.train(train_dir = 'C:/Users/Etienne.Houze/Documents/EtienneData/trainmediumlesslabs', log_dir='log_day16/1', batch_size = 5, train_size = 10000, epochs = 100, learning_rate = 1e-4, trainable_vars = ['32s'],savestep=500)

model_test.test(testdir = 'C:/Users/Etienne.Houze/Documents/EtienneData/valless/valless',num_im=100)

#model_test.compute_and_save(imlist = ['C:/Users/Etienne.Houze/Documents/EtienneData/valless/valless/_4_im_.png'], orig_imH = 256, orig_imW = 512, outdir = 'C:/Users/Etienne.Houze/Documents/EtienneData/outdir')