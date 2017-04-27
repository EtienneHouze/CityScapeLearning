from __future__ import print_function, division, absolute_import

import tensorflow as tf
from Network import *
from CityScapeLearning import *
from PIL import Image
import helpers
import labels
from os.path import join

class Model:
    """
        Describes a model.
        Fields :
            - saving_dir : path to the folder where the model will be saved during training
            - name : a name for the model
            - netbuilder : un function from Network to build the net, see Network class documentation.
            - imH, imW : height and width of the input images of the network of the model
            - num_labs : how many labels are classified by this model. Can be 35 or 8
            - last_cp : path to the file of the last checkpoint of the model
            - trained_vars : list of variables which has already been learned by the net and wont be initialized again but restored from the checkpoint instead.
    """

    def __init__(self, name, saving_dir, netbuilder, imH, imW, num_labs, last_cp = None, trained_vars = []):
       self.saving_dir = saving_dir
       self.name = name
       self.netbuilder = netbuilder
       self.imH = imH
       self.imW = imW
       self.num_labs = num_labs
       self.last_cp = last_cp
       self.trained_vars = trained_vars


    def train(self, train_dir, log_dir, batch_size, epochs, train_size, learning_rate, savestep = 100, trainable_vars = None):
        """
            Trains the model.
            @ args  :
                - train_dir : folder containing training images
                - log_dir : path to the tensorboard log
                - batch_size : length of a mini batch
                - epochs : number of epochs to perform
                - train_size : number of images to use in the training set
                - learning_rate : learning rate of the optimizer
                - savestep : how often a checkpoint is recorded
                - trainable vars : list of variables to train. If none, all are learned. See documentation of the net for further details.
        """
        num_labs = self.num_labs
        imH = self.imH
        imW = self.imW
        train_set, histo = helpers.produce_training_set(traindir=train_dir,trainsize=train_size,numlabs=num_labs)
        freqs = histo/np.sum(histo)
        mainGraph = tf.Graph()
        with mainGraph.as_default():
            with tf.name_scope('Input'):
                ins = tf.placeholder(shape=(batch_size, self.imH, self.imW, 3),
                                        dtype=tf.float32)
                labs = tf.placeholder(shape=(batch_size, self.imH, self.imW),
                                        dtype=tf.int32)
                weigs = tf.placeholder(shape=(num_labs),
                                        dtype=tf.float32)

            with tf.name_scope("Net"):
                CNN = self.netbuilder(input=ins,numlab=num_labs)
            global_step = tf.Variable(initial_value=0,
                                        name='global_step',
                                        trainable=False)
            with tf.name_scope('inputs'):
                helpers.image_summaries(ins)
            with tf.name_scope('out'):
                helpers.image_summaries(tf.expand_dims(input=CNN.output, axis=-1), name='output')
                helpers.variable_summaries(tf.cast(CNN.output, dtype=tf.float32))
            with tf.name_scope('labels'):
                helpers.image_summaries(tf.expand_dims(input=labs, axis=-1), name='labels')
                helpers.variable_summaries(tf.cast(labs, dtype=tf.float32))

            with tf.name_scope('Loss'):
                l = perso_loss(logits=CNN.last_layer, labs=labs, weights=weigs)
                tf.summary.scalar(name='loss', tensor=l)

            with tf.name_scope('Learning'):
                trainstep = None
                if (trainable_vars):
                    varlist = []
                    self.trained_vars.extend(trainable_vars)
                    self.trained_vars = list(set(self.trained_vars))
                    for varname in trainable_vars:
                        varlist.extend(CNN.variables[varname])
                    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=l,                                                                global_step=global_step,
                                                                                              var_list = varlist)
                else:
                    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=l,                                                                global_step=global_step)


            merged = tf.summary.merge_all()
    
        with tf.Session(graph=mainGraph) as sess:

            trainWriter = tf.summary.FileWriter(logdir=log_dir, graph=sess.graph)
            sess.run(tf.global_variables_initializer())

            if (self.last_cp and len(self.trained_vars) != 0):
                trained_var_list = [global_step]
                for var in self.trained_vars:
                    trained_var_list.extend(CNN.variables[var])
                loader = tf.train.Saver(trained_var_list)
                loader.restore(sess,self.last_cp)
            elif (self.last_cp):
                loader = tf.train.Saver()
                loader.restore(sess,self.last_cp)
            for epoch in range(epochs):
                random.shuffle(train_set)
                for i in range(int(train_size / batch_size)):
                    if (i == 0):
                        [images, labels, w] = helpers.produce_mini_batch(train_set, step=i, imW=imW, imH=imH, batch_size=batch_size, numlabs=num_labs)
                        _, out, test_layer, test_loss, summary, step = sess.run(
                            (train_step, CNN.output, CNN.last_layer, l, merged, global_step),
                            feed_dict={ins: images, labs: labels, weigs : freqs})
                        print( 'At step ' + str(i) + ' in epoch ' + str(epoch) +', loss is ' +str(test_loss))
                        trainWriter.add_summary(summary, step)
                    else:
                        [images, labels, w] = helpers.produce_mini_batch(train_set, step=i, imW=imW, imH=imH, batch_size=batch_size, numlabs = num_labs)
                        _, out, test_layer, test_loss, summary, step = sess.run(
                            (train_step, CNN.output, CNN.last_layer, l, merged, global_step),
                            feed_dict={ins: images, labs: labels, weigs : freqs})
                        print( 'At step ' + str(i) + ' in epoch ' + str(epoch) +', loss is ' +str(test_loss))
                        trainWriter.add_summary(summary, step)
                    if (step % savestep == 0):
                        saver = tf.train.Saver()
                        self.last_cp = saver.save(sess,
                                                  save_path = join(self.saving_dir,self.name),
                                                  global_step = step,
                                                  write_meta_graph = False
                                                  )


    def test(self, testdir, num_im = 100):
        print("Testing the model...")
        print("---------------")
        print("Building the graph")
        imH = self.imH
        imW = self.imW
        batch_size = 1
        #Building the model
        mainGraph = tf.Graph()
        with mainGraph.as_default():
            with tf.name_scope('Input'):
                ins = tf.placeholder(shape=(batch_size, imH, imW, 3),
                                        dtype=tf.float32)
                labs = tf.placeholder(shape=(batch_size, imH, imW),
                                        dtype=tf.int32)
                weigs = tf.placeholder(shape=(batch_size,self.num_labs),
                                        dtype=tf.float32)

            with tf.name_scope("Net"):
                CNN = self.netbuilder(input=ins,numlab=self.num_labs)
            with tf.name_scope('Loss'):
                l = perso_loss(logits=CNN.last_layer, labs=labs, weights=weigs)
        print("Done !")
        
        with tf.Session(graph = mainGraph) as sess:
            print("Restoring variables")
            sess.run(tf.global_variables_initializer())
            test_set = helpers.produce_testing_set(testdir, num_im,imH=imH,imW = imW)
            if (self.last_cp and len(self.trained_vars) != 0):
                trained_var_list = []
                for var in self.trained_vars:
                    trained_var_list.extend(CNN.variables[var])
                loader = tf.train.Saver(trained_var_list)
                loader.restore(sess,self.last_cp)
            elif (self.last_cp):
                loader = tf.train.Saver()
                loader.restore(sess,self.last_cp)
            tot_IOU = np.zeros((num_im))
            tot_acc = np.zeros((num_im))
            print("Done ! \n")
            for i in range(num_im):
                [images, labels, w] = helpers.produce_mini_batch(test_set, step=i, imW=imW, imH=imH, batch_size=batch_size, numlabs=self.num_labs)
                preds, test_loss = sess.run((CNN.output,l),feed_dict = {ins : images, labs : labels, weigs : w})
                print("Image "+str(i)+" :")
                print("==============================")
                print("     Loss is : " + str(test_loss))
                IOU = np.zeros((self.num_labs))
                acc = np.zeros((self.num_labs))
                for lab_ind in range(self.num_labs):
                    TP = 0.0
                    FP = 0.0
                    FN = 0.0
                    TN = 0.0
                    for j in range(imH):
                        for k in range(imW):
                            if (preds[0,j,k]==lab_ind and labels[0][j,k]==lab_ind):
                                TP += 1
                            elif (preds[0,j,k]==lab_ind and labels[0][j,k]!=lab_ind):
                                FP += 1
                            elif (preds[0,j,k]!=lab_ind and labels[0][j,k]==lab_ind):
                                FN +=1
                            else :
                                TN += 1
                    IOU[lab_ind] = TP/(TP+FP+FN)
                    acc[lab_ind] = (TP + TN) / (TP+TN+FP+FN)
                    print('For label' + str(lab_ind) + ', IOU is ' +str(IOU[lab_ind]))
                IOU_mean = np.mean(IOU)
                acc_mean = np.mean(acc)
                tot_IOU[i] = IOU_mean
                tot_acc[i] = acc_mean
                print('     mean IOU is : ' + str(IOU_mean))
                print('     mean accuracy is : ' + str(acc_mean))
            print(' ')
            print("Over " + str(num_im) + " images, mean IOU is " + str(np.mean(tot_IOU))+ ", mean accuracy is : " + str(np.mean(tot_acc)))
            print("End")
                

    def compute(self, im):
        print("Using the model...")
        print("---------------")
        print("Building the graph")
        imH = self.imH
        imW = self.imW
        batch_size = 1
        #Building the model
        mainGraph = tf.Graph()
        with mainGraph.as_default():
            with tf.name_scope('Input'):
                ins = tf.placeholder(shape=(batch_size, imH, imW, 3),
                                        dtype=tf.float32)
                labs = tf.placeholder(shape=(batch_size, imH, imW),
                                        dtype=tf.int32)
                weigs = tf.placeholder(shape=(batch_size,self.num_labs),
                                        dtype=tf.float32)

            with tf.name_scope("Net"):
                CNN = self.netbuilder(input=ins,numlab=self.num_labs)
            with tf.name_scope('Loss'):
                l = perso_loss(logits=CNN.last_layer, labs=labs, weights=weigs)
        print("Done !")

        with tf.Session(graph = mainGraph) as sess:
            print("Restoring variables")
            sess.run(tf.global_variables_initializer())
            if (self.last_cp and len(self.trained_vars) != 0):
                trained_var_list = []
                for var in self.trained_vars:
                    trained_var_list.extend(CNN.variables[var])
                loader = tf.train.Saver(trained_var_list)
                loader.restore(sess,self.last_cp)
            elif (self.last_cp):
                loader = tf.train.Saver()
                loader.restore(sess,self.last_cp)
            print("Done ! \n")
            Im = Image.open(im)
            Im.thumbnail((imW,imH))
            im_array = np.expand_dims(np.asarray(a = Im,
                                                 dtype = np.float32
                                                 ),
                                      axis = 0
                                      )
            out = sess.run(CNN.output, feed_dict = {ins : im_array})
            out_view = np.zeros(shape=(out.shape[1], out.shape[2], 3), dtype = np.uint8)
            for i in range(out.shape[1]):
                for j in range(out.shape[2]):
                    lab = labels.id2label[out[0,i,j]]
                    out_view[i, j, :] = lab.color
            Out = Image.fromarray(out_view)
            Out.show()
            print("test")