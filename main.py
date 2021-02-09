# -*- coding:utf-8 -*-
# Created on 12.22 2020
# @author: HengLi

import os
import argparse

from model import Adversarial_Reprogramming as AR
import os
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # 使用第二块GPU（从0开始）

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--network_name', dest='network_name', default='inception_resnet_v2', help='inception_resnet_v2,inception_v3,inception_v4,resnet_v2_50,resnet_v2_101,resnet_v2_152,vgg_16,vgg_19,mobilenet_v1_128')
parser.add_argument('--image_size', dest='image_size', type=int, default=224, help='size of input images')

""" Arguments related to dataset"""
parser.add_argument('--dataset', dest='dataset', default='app', help='app, mnist, f_mnist')
# parser.add_argument('--central_size', dest='central_size', type=int, default=28, help='28 for MNIST,32 for CIFAR10')

"""Arguments related to run mode"""
#parser.add_argument('--restore', dest='restore', default=None, action='store', type=int, help='Specify checkpoint id to restore.')

"""Arguments related to training"""
parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=40, help='max num of epoch')
parser.add_argument('--lr', dest='lr', type=float, default=0.003, help='initial learning rate for adam')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=5, help='save frequence')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=50, help='# images in batch')
parser.add_argument('--lmd', dest='lmd', type=float, default=2e-6, help='# weights of norm penalty')#0.01
parser.add_argument('--decay', type=float, default=0.96, help='Decay to apply to lr')

"""Arguments related to monitoring and outputs"""
parser.add_argument('--result_dir', dest='result_dir', default='./train', help='train logs are saved here')
parser.add_argument('--model_dir', dest='model_dir', default='./model', help='pretrained models are saved here')
parser.add_argument('--data_dir', dest='data_dir', default='./datasets', help='datasets are stored here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='datasets are stored here')
parser.add_argument('--mapping', dest='mapping_mode', type=int, default=1, help='1,2,3 the number of the layer of the mapping module')

args = parser.parse_args()
print(args)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    args.model_dir = "./model/" + args.network_name
    args.data_dir = "./datasets/" + args.dataset
    args.result_dir = "./result/" + args.dataset + "/" + args.network_name + "/" + str(args.mapping_mode)
    args.sample_dir = args.result_dir + "/sample"
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)


    model = AR(args)
    model.run()

if __name__ == '__main__':
    main()    

    


   