import CityScapeLearning as city
import helpers
from Network import *
from Model import Model


model_test = Model(name='model_test',saving_dir='D:/EtienneData/models/day_11/2', netbuilder = build_big_CNN_2skips,imH=256,imW=512,num_labs=8,last_cp='D:/EtienneData/models/day_11/2/model_test-300')

model_test.train(train_dir = 'D:/EtienneData/trainmediumlesslabs', log_dir='log_day11/2', batch_size = 10, train_size = 10000, epochs = 20, learning_rate = 1e-4, vars = ['32s'])

model_test.test(testdir = 'D:/EtienneData/valless',num_im=10)
