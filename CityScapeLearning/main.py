import CityScapeLearning as city
import helpers

#city.train(train_dir = 'D:/EtienneData/trainsmalllesslabs',log_dir='log_day8/4',batch_size=10,epochs=100,train_size=2000,learning_rate=1e-4,imH=128,imW=256,num__labs=8)

helpers.produce_training_dir(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',
                             labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',
                             crop=False,
                             outdir = 'D:/EtienneData/trainsmalllesslabs',
                             training_set_size=10000,
                             imH = 128,
                             imW = 256,
                             alllabels=False
                             )

helpers.produce_training_dir(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',
                             labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/train',
                             crop=False,
                             outdir = 'D:/EtienneData/trainsmediumlesslabs',
                             training_set_size=10000,
                             imH = 256,
                             imW = 512,
                             alllabels=False
                             )
