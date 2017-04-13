import CityScapeLearning as city
import helpers

city.train(train_dir = 'D:/EtienneData/smalltrainresized',log_dir='log_day7/13',batch_size=10,epochs=10,train_size=2000,learning_rate=5e-4,imH=128,imW=256)

#helpers.produce_training_dir(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',
#                             labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',
#                             crop=False,
#                             outdir = 'D:/EtienneData/train256512',
#                             training_set_size=2000,
#                             imH = 256,
#                             imW = 512
                             #)