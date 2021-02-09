# DuENNA: Towards Securing DNN Model Deployment
TensorFlow implementation of DuENNA(KDD2021-219)

## Setup

### Prerequisites
- Python 3.6 
- TensorFlow 1.14 
- numpy
- tf_slim
- scipy 1.2.1
- pillow

### Download the imagenet models
Download the following pre-trained models https://github.com/tensorflow/models/tree/master/research/slim
Put them into './model/model_name':

- [resnet_v2_50]
- [resnet_v2_101]
- [resnet_v2_152]
- [inception_v3]
- [inception_resnet_v2]
- [inception_v4]
- [inception_v5]
- [vgg_16]
- [vgg_19]
- [MobileNet_v1_0.25]
- [MobileNet_v1_0.5]


### Datasets
- MNIST, FASHION-MNIST and ANDROIDZOO dateset have been saved in 'dataset' fold

## Train
Simply run the following command to show an example:
```
python main.py --network_name mobilenet_v1_128 --dataset mnist --mapping 1
```
To be specific:
'network_name' is the pre-trained models and it can be chosen from
['inception_resnet_v2','inception_v3','inception_v4','resnet_v2_50','resnet_v2_101','resnet_v2_152','vgg_16','vgg_19','mobilenet_v1_128','mobilenet_v1_64'] 

'mapping' is the number of the back-end model's layer, it can be chosen from [1,2,3]

'dataset' is the task name for we want to train, it can be chosen from ['mnist','f_mnist','app']

## Results
The performance of the model will be shown in every epoch, and some results will be saved in 'result'


## Acknowledgments
Code referred to  [Pytorch implementation](https://github.com/Prinsphield/Adversarial_Reprogramming). 
