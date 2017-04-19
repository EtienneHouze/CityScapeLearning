import CityScapeLearning as city
import helpers

city.train(train_dir = 'D:/EtienneData/trainsmalllesslabs',log_dir='log_day9/7',batch_size=10,epochs=20,train_size=1000,learning_rate=1e-4,imH=128,imW=256,num__labs=8, saving_path ='D:/EtienneData/models/day_9ter', loading_path=None)

city.test('D:/EtienneData/valless',savedmodel = 'D:/EtienneData/models/day_9ter', num__labs = 8, num_im = 5)

#helpers.produce_training_dir(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val',
#                             labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/val',
#                             crop=False,
#                             outdir = 'D:/EtienneData/valless',
#                             training_set_size=1000,
#                             imH = 1024,
#                             imW = 2048,
#                             alllabels=False
#                             )

#helpers.produce_training_dir(imdir='D:/EtienneData/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val',
#                             labeldir='D:/EtienneData/Cityscapes/gtFine_trainvaltest/gtFine/val',
#                             crop=False,
#                             outdir = 'D:/EtienneData/valall',
#                             training_set_size=1000,
#                             imH = 1024,
#                             imW = 2048,
#                             alllabels=True
#                             )
