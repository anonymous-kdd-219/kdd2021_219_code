# -*- coding: utf-8 -*-
# Created on Tue Aug  7 10:29:59 2018
# @author: Zhuorong Li  <lizr@zucc.edu.cn>

# To get pre-trained models used in this repository:
# wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
# wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
# More pre-trained models can be found: https://github.com/tensorflow/models/tree/master/research/slim

# MNIST dataset will be automatically downloaded after running the scripts.

import cv2
import numpy as np
import os
import random
import scipy.misc
from time import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
from nets import inception_resnet_v2,inception_v3,inception_v4,resnet_v2,resnet_utils,vgg,mobilenet_v1_128
from glob import glob

####网络打表####
network ={"inception_resnet_v2":inception_resnet_v2.inception_resnet_v2,"inception_v3":inception_v3.inception_v3,
"inception_v4":inception_v4.inception_v4,
"resnet_v2_50":resnet_v2.resnet_v2_50,"resnet_v2_101":resnet_v2.resnet_v2_101,"resnet_v2_152":resnet_v2.resnet_v2_152,
"vgg_16":vgg.vgg_16,"vgg_19":vgg.vgg_19,"mobilenet_v1_64":mobilenet_v1_128.mobilenet_v1_025,"mobilenet_v1_128":mobilenet_v1_128.mobilenet_v1_050}


network_scope ={"inception_resnet_v2":inception_resnet_v2.inception_resnet_v2_arg_scope,"inception_v3":inception_v3.inception_v3_arg_scope,
"inception_v4":inception_v4.inception_v4_arg_scope,
"resnet_v2_50":resnet_utils.resnet_arg_scope,"resnet_v2_101":resnet_utils.resnet_arg_scope,"resnet_v2_152":resnet_utils.resnet_arg_scope,
"vgg_16":vgg.vgg_arg_scope,"vgg_19":vgg.vgg_arg_scope,"mobilenet_v1_64":mobilenet_v1_128.mobilenet_v1_arg_scope,"mobilenet_v1_128":mobilenet_v1_128.mobilenet_v1_arg_scope}

network_scope_name ={"inception_resnet_v2":"InceptionResnetV2","inception_v3":"InceptionV3",
"inception_v4":"InceptionV4",
"resnet_v2_50":"resnet_v2_50","resnet_v2_101":"resnet_v2_101","resnet_v2_152":"resnet_v2_152",
"vgg_16":"vgg_16","vgg_19":"vgg_19","mobilenet_v1_64":"MobilenetV1","mobilenet_v1_128":"MobilenetV1"}


# 读取目录下所有的jpg图片
def load_image(image_path, image_size=64):
    file_name=glob(image_path+"/*JPEG")
    sample = []
    for file in file_name:
        pic = scipy.misc.imread(file).astype(np.float32)
        pic = scipy.misc.imresize(pic, (image_size, image_size)).astype(np.float32)
        pic = np.array(pic)
        if pic.shape == (64,64,3):
            sample.append(pic)

    sample = np.array(sample)/127.5-1
    return sample

class Inputs(object):
    def __init__(self):
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets('./datasets/f_mnist', one_hot=True)
 
    def load_train_data_and_labels(self):
        train_x = self.mnist.train.images
        train_y = self.mnist.train.labels
        return train_x, train_y
 
    def load_eval_data_and_labels(self):
        eval_x = self.mnist.test.images
        eval_y = self.mnist.test.labels
        return eval_x, eval_y



class Adversarial_Reprogramming(object):
    def __init__(self, args):
        
        self.network_name=args.network_name
        self.dataset=args.dataset
        # self.central_size=args.central_size
        self.image_size=args.image_size
        self.max_epoch=args.max_epoch
        self.lr=args.lr
        self.batch_size=args.batch_size
        self.lmd=args.lmd
        self.decay=args.decay
        self.save_freq=args.save_freq
        self.result_dir=args.result_dir
        self.model_dir=args.model_dir
        self.data_dir=args.data_dir
        self.sample_dir=args.sample_dir
        self.mapping_mode = args.mapping_mode
        # print(self.dataset)
        # input()
        if self.dataset == 'mnist':
            mnist = input_data.read_data_sets('./datasets/mnist', one_hot=True)
            test_images = mnist.test.images 
            self.x_test = np.reshape(test_images, [-1, 28, 28, 1])
            self.y_test = mnist.test.labels
            self.test_data_length = len(mnist.test.images)

            train_images = mnist.train.images 
            self.x_train = np.reshape(train_images, [-1, 28, 28, 1])
            self.y_train = mnist.train.labels
            self.train_data_length = len(mnist.train.images)
            self.central_size = 28

        elif self.dataset == 'app':
            app_dataset = np.array(np.load('./datasets/app/dataset_norepeat.npy'), dtype=np.float32)  
            count_0 = 0
            count_1 = 0
            begin = 20000
            for i in app_dataset:
                if i[-1]==0:
                    count_0+=1
                if i[-1]==1:
                    count_1+=1

            index = [i for i in range(app_dataset.shape[0])] 
            random.shuffle(index)
            app_dataset = app_dataset[index]
            x_train = app_dataset[:begin, 0:379]

            self.x_train = np.reshape(np.concatenate((app_dataset[:begin, 0:379],np.zeros([begin, 21])),axis=1),(begin,20,20,1))*2-1
            self.y_train = app_dataset[:begin, 379:]

            self.x_test = np.reshape(np.concatenate((app_dataset[begin:, 0:379],np.zeros([app_dataset[begin:, 0:379].shape[0], 21])),axis=1),(-1,20,20,1))*2-1
            self.y_test = app_dataset[begin:, 379:]
            self.train_data_length = self.x_train.shape[0]
            self.test_data_length = self.x_test.shape[0]
            self.central_size = 20
            
        elif self.dataset == 'f_mnist':
        
            inputs = Inputs()
            self.x_train, self.y_train = inputs.load_train_data_and_labels()
            self.x_test, self.y_test = inputs.load_eval_data_and_labels()
            self.train_data_length = self.x_train.shape[0]
            self.test_data_length = self.x_test.shape[0]
            self.central_size = 28
        else:
            print("error dataset name")
            return 0

    def label_mapping(self):
            imagenet_label = np.zeros([1001,10])
            imagenet_label[0:10,0:10]=np.eye(10)
            return tf.constant(imagenet_label, dtype=tf.float32)  
    
    def adv_program(self,central_image):
        if self.network_name.startswith('inception'):
            self.image_size = 299
            means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.network_name.startswith('mobilenet_v1_64'):
            self.image_size = 64
            means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.network_name.startswith('mobilenet_v1_128'):
            self.image_size = 128
            means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            self.image_size = self.image_size
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
        M = np.pad(np.zeros([1, self.central_size, self.central_size, 3]),\
                           [[0,0], [int((np.ceil(self.image_size/2.))-self.central_size/2.), int((np.floor(self.image_size/2.))-self.central_size/2.)],\
                            [int((np.ceil(self.image_size/2.))-self.central_size/2.), int((np.floor(self.image_size/2.))-self.central_size/2.)],\
                             [0,0]],'constant', constant_values = 1)
        # c in (-1,1)
        C = scipy.misc.imread("./cover/1.jpg",mode="RGB")      
        C = scipy.misc.imresize(C,(self.image_size,self.image_size))
        C = C.reshape(1,self.image_size,self.image_size,3)/127.5-1




        self.M = tf.constant(M, dtype=tf.float32)
        self.C = tf.constant(C, dtype=tf.float32)
        with tf.variable_scope('adv_program',reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('program',shape=[1,self.image_size,self.image_size,3], dtype = tf.float32)


            if self.network_name.startswith('vgg'):
                if self.dataset == 'app':
                    if self.mapping_mode == 1:
                        self.mapping = tf.get_variable('program9',shape=[1000,2], dtype = tf.float32)
                    elif self.mapping_mode == 2:
                        self.mapping = tf.get_variable('program9',shape=[1000,200], dtype = tf.float32)
                        self.mapping1 = tf.get_variable('program10',shape=[200,2], dtype = tf.float32)
                    elif self.mapping_mode == 3:
                        self.mapping = tf.get_variable('program9',shape=[1000,200], dtype = tf.float32)
                        self.mapping1 = tf.get_variable('program10',shape=[200,100], dtype = tf.float32)
                        self.mapping2 = tf.get_variable('program11',shape=[100,2], dtype = tf.float32)

                else:
                    if self.mapping_mode == 1:
                        self.mapping = tf.get_variable('program9',shape=[1000,10], dtype = tf.float32)
                    elif self.mapping_mode == 2:
                        self.mapping = tf.get_variable('program9',shape=[1000,200], dtype = tf.float32)
                        self.mapping1 = tf.get_variable('program10',shape=[200,10], dtype = tf.float32)
                    elif self.mapping_mode == 3:
                        self.mapping = tf.get_variable('program9',shape=[1000,200], dtype = tf.float32)
                        self.mapping1 = tf.get_variable('program10',shape=[200,100], dtype = tf.float32)
                        self.mapping2 = tf.get_variable('program11',shape=[100,10], dtype = tf.float32)
            else:
                if self.dataset == 'app':
                    if self.mapping_mode == 1:
                        self.mapping = tf.get_variable('program9',shape=[1001,2], dtype = tf.float32)
                    elif self.mapping_mode == 2:
                        self.mapping = tf.get_variable('program9',shape=[1001,200], dtype = tf.float32)
                        self.mapping1 = tf.get_variable('program10',shape=[200,2], dtype = tf.float32)
                    elif self.mapping_mode == 3:
                        self.mapping = tf.get_variable('program9',shape=[1001,200], dtype = tf.float32)
                        self.mapping1 = tf.get_variable('program10',shape=[200,100], dtype = tf.float32)
                        self.mapping2 = tf.get_variable('program11',shape=[100,2], dtype = tf.float32)

                else:
                    if self.mapping_mode == 1:
                        self.mapping = tf.get_variable('program9',shape=[1001,10], dtype = tf.float32)
                    elif self.mapping_mode == 2:
                        self.mapping = tf.get_variable('program9',shape=[1001,200], dtype = tf.float32)
                        self.mapping1 = tf.get_variable('program10',shape=[200,10], dtype = tf.float32)
                    elif self.mapping_mode == 3:
                        self.mapping = tf.get_variable('program9',shape=[1001,200], dtype = tf.float32)
                        self.mapping1 = tf.get_variable('program10',shape=[200,100], dtype = tf.float32)
                        self.mapping2 = tf.get_variable('program11',shape=[100,10], dtype = tf.float32)


        if self.dataset == 'app':
           central_image  = tf.concat([central_image, central_image, central_image], axis = -1) *2-1
        elif self.dataset == 'mnist':
           central_image  = tf.concat([central_image, central_image, central_image], axis = -1) *2-1
        elif self.dataset == 'f_mnist':
           central_image  = tf.concat([central_image, central_image, central_image], axis = -1) *2-1


        self.X = tf.pad(central_image,
                    paddings = tf.constant([[0,0], [int((np.ceil(self.image_size/2.))-self.central_size/2.), int((np.floor(self.image_size/2.))-self.central_size/2.)],\
                             [int((np.ceil(self.image_size/2.))-self.central_size/2.), int((np.floor(self.image_size/2.))-self.central_size/2.)], [0,0]]))

        #扰动去掉中心的地方
        self.P = tf.nn.tanh(tf.multiply(self.W, self.M))
        self.C = tf.multiply(self.C, self.M)
        #加上嵌入的图片
        self.X_adv = self.X + self.P+self.C 

        
        return self.X_adv
    
    def run(self):
        if self.dataset == 'app':
            input_images  = tf.placeholder(shape = [None, 20,20,1], dtype = tf.float32)
            Y = tf.placeholder(tf.float32, shape=[None, 2]) 
        else:
            input_images  = tf.placeholder(shape = [None, self.central_size,self.central_size,1], dtype = tf.float32)
            Y = tf.placeholder(tf.float32, shape=[None, 10]) 
        


        with slim.arg_scope(network_scope[self.network_name]()):
            if self.network_name.startswith('vgg'):            
                self.imagenet_logits,_ = network[self.network_name](self.adv_program(input_images), num_classes = 1000,is_training=False)
            else:
                self.imagenet_logits,_ = network[self.network_name](self.adv_program(input_images), num_classes = 1001,is_training=False)               

            if self.network_name.startswith('mobilenet_v1_64'):            
                model_path = tf.train.latest_checkpoint(self.model_dir)
                model_path = "./model/mobilenet_v1_64/mobilenet_v1_0.25_128.ckpt"
            elif self.network_name.startswith('mobilenet_v1_128'):            
                model_path = tf.train.latest_checkpoint(self.model_dir)
                model_path = "./model/mobilenet_v1_128/mobilenet_v1_0.5_160.ckpt"
            else:
                files = os.listdir(self.model_dir)
                model_path = [self.model_dir +'/'+ f for f in files if f.endswith(('.ckpt'))][0]

            if self.mapping_mode == 1:
                self.disturbed_logits = tf.matmul(self.imagenet_logits,self.mapping)
            elif self.mapping_mode == 2:
                self.disturbed_logits = tf.matmul(self.imagenet_logits,self.mapping)
                self.disturbed_logits = tf.matmul(self.disturbed_logits,self.mapping1) 
            elif self.mapping_mode == 3: 
                self.disturbed_logits = tf.matmul(self.imagenet_logits,self.mapping)
                self.disturbed_logits = tf.matmul(self.disturbed_logits,self.mapping1)
                self.disturbed_logits = tf.matmul(self.disturbed_logits,self.mapping2)
            init_fn = slim.assign_from_checkpoint_fn(model_path,slim.get_model_variables(network_scope_name[self.network_name]))


        ## loss function
        self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits = self.disturbed_logits))
        self.reg_loss = self.lmd * (tf.nn.l2_loss(self.W))
        self.perturbation  = tf.reduce_mean(tf.square(self.P))
        self.loss = self.cross_entropy_loss + self.reg_loss#+self.perturbation


        ## compute accuracy
        correct_prediction = tf.equal(tf.argmax(self.disturbed_logits,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
       
        ## optimizer
        global_steps = tf.Variable(0, trainable=False)
        initial_learning_rate = self.lr
        steps_per_epoch = int(self.train_data_length/ self.batch_size)
        decay_steps = 2 * steps_per_epoch
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_steps, decay_steps, self.decay, staircase=True)
        optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(self.loss,var_list = [self.W],global_step=global_steps) 
        
        if self.mapping_mode == 1:
            optimizer1 = tf.train.AdamOptimizer(initial_learning_rate/5).minimize(self.loss,var_list = [self.mapping],global_step=global_steps) 
        elif self.mapping_mode == 2:
            optimizer1 = tf.train.AdamOptimizer(initial_learning_rate/5).minimize(self.loss,var_list = [self.mapping,self.mapping1],global_step=global_steps) 
        elif self.mapping_mode == 3: 
            optimizer1 = tf.train.AdamOptimizer(initial_learning_rate).minimize(self.loss,var_list = [self.mapping,self.mapping1,self.mapping2],global_step=global_steps) 

        ## Training
        vl = [v for v in tf.global_variables() if "Adam" not in v.name and "beta" not in v.name]
        saver = tf.train.Saver(var_list=vl)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        init_fn(sess)

        ########test set
        # print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.all_variables()]))
        # input()
        
        
        if self.dataset == 'app':        
            test_images = self.x_test
            test_images = np.reshape(test_images, [-1, 20, 20, 1])
            test_labels =self.y_test
            test_total_batch = int(self.test_data_length/self.batch_size)
        else:
            test_images = self.x_test
            test_images = np.reshape(test_images, [-1, 28, 28, 1])
            test_labels =self.y_test
            test_total_batch = int(self.test_data_length/self.batch_size)




        ## restore if checkpoint flies exist
        # ckpt = tf.train.get_checkpoint_state(self.train_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess,ckpt.model_checkpoint_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        total_batch = int(self.train_data_length/self.batch_size)
        training_start = time()
        for epoch in range(self.max_epoch):
            epoch_start=time()
            for batch in range(total_batch):

                image_batch = self.x_train[self.batch_size*batch:self.batch_size*(batch+1)]
                label_batch = self.y_train[self.batch_size*batch:self.batch_size*(batch+1)]
                if self.dataset == 'app':        
                    image_batch = np.reshape(image_batch, [-1, 20, 20, 1])
                else:
                    image_batch = np.reshape(image_batch, [-1, 28, 28, 1])
                _, W,mapping,train_loss, img_X_adv,X,P,C = sess.run([optimizer,self.W, self.mapping,self.loss, self.X_adv,self.X , self.P,self.C ],\
                                         feed_dict = {input_images:image_batch, Y:label_batch})



                _, W,mapping,train_loss, img_X_adv,X,P,C = sess.run([optimizer1,self.W, self.mapping,self.loss, self.X_adv,self.X , self.P,self.C ],\
                                         feed_dict = {input_images:image_batch, Y:label_batch})


                print('train_loss:{:.4f}'.format(train_loss),"epoch:",epoch,"batch:",batch,"total_batch:",total_batch) 

            test_acc_sum = 0
            for i in range(test_total_batch):
                test_image_batch = test_images[i*self.batch_size:(i+1)*self.batch_size]
                test_label_batch = test_labels[i*self.batch_size:(i+1)*self.batch_size]
                test_batch_acc = sess.run(accuracy, feed_dict = {input_images:test_image_batch,Y:test_label_batch})
                test_acc_sum += test_batch_acc
            test_acc = str(float(test_acc_sum/test_total_batch))
            acc_dir = self.result_dir + '/test_acc.txt'
            with open(acc_dir,'a') as file_handle:   
                file_handle.write(test_acc+" ")   
                file_handle.write('\n')    

            print('test_acc:{:.4f}'.format(float(test_acc_sum/test_total_batch)),"################") 

            if (epoch+1) % self.save_freq == 0:
                # saver.save(sess, os.path.join(self.train_dir, 'model_{:06d}.ckpt'.format(epoch+1)))
                for j in range(5):                    
                    scipy.misc.toimage(np.clip(img_X_adv[j],-1,1)/2+0.5).save(os.path.join(self.sample_dir,'epoch_{:06d}_{}.jpg'.format((epoch+1),j))) 
                    scipy.misc.toimage(np.clip(X[j],-1,1)/2+0.5).save(os.path.join(self.sample_dir,'X{:06d}_{}.jpg'.format((epoch+1),j))) 
                    scipy.misc.toimage(np.clip(P[0],-1,1)/2+0.5).save(os.path.join(self.sample_dir,'P{:06d}_{}.jpg'.format((epoch+1),j))) 
                    scipy.misc.toimage(np.clip(C[0],-1,1)/2+0.5).save(os.path.join(self.sample_dir,'C{:06d}_{}.jpg'.format((epoch+1),j))) 



            test_acc = float(test_acc_sum/test_total_batch)
            print('test_acc:{:.4f}'.format(test_acc),"epoch:",epoch) 

            epoch_duration =time()-epoch_start
            print("Training this epoch takes:","{:.2f}".format(epoch_duration))
        training_duration = time()-training_start

        ## Test when training finished
        testing_start = time()

        test_acc_sum = 0.0
        for i in range(test_total_batch):
            test_image_batch = test_images[i*self.batch_size:(i+1)*self.batch_size]
            test_label_batch = test_labels[i*self.batch_size:(i+1)*self.batch_size]
            test_batch_acc = sess.run(accuracy, feed_dict = {input_images:test_image_batch,Y:test_label_batch})
            test_acc_sum += test_batch_acc
        test_acc = float(test_acc_sum/test_total_batch)
        testing_duration = time()-testing_start
        print('test_acc:{:.4f}'.format(test_acc)) 
        print("Training {:03d}".format(self.max_epoch)+" epoches takes:{:.2f} secs".format(training_duration))    
        print("Testing finished takes:{:.2f} secs".format(testing_duration))  

        ckp_dir = self.result_dir + '/checkpoint'
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)

        if self.mapping_mode == 1:                 
            W,mapping = sess.run([self.W, self.mapping] )
            np.save(ckp_dir + "/mapping.npy", mapping)
        elif self.mapping_mode == 2:
            W,mapping,mapping1 = sess.run([self.W, self.mapping, self.mapping1] )
            np.save(ckp_dir + "/mapping.npy", mapping)
            np.save(ckp_dir + "/mapping1.npy", mapping1)
        elif self.mapping_mode == 3: 
            W,mapping,mapping1,mapping2 = sess.run([self.W, self.mapping, self.mapping1, self.mapping2] )
            np.save(ckp_dir + "/mapping.npy", mapping)
            np.save(ckp_dir + "/mapping1.npy", mapping1)
            np.save(ckp_dir + "/mapping2.npy", mapping2)

        scipy.misc.toimage(W[0]).save(os.path.join(ckp_dir,'Wepoch_{:06d}.jpg'.format((epoch+1)))) 
        coord.join(threads)

                    
            
            
        

                             
