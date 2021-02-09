import argparse
import collections
from inspect import currentframe
import os

from easydict import EasyDict as edict

frame = currentframe().f_back
while frame.f_code.co_filename.startswith('<frozen'):
    frame = frame.f_back
import_from = frame.f_code.co_filename
eval_mode = 0 if 'eval' not in import_from else 1

config = edict(d=collections.OrderedDict())
# attack related
config.attack_network = "0123459a"
config.method = "3.0_msss_4"
config.step_size = 3.2
config.max_epsilon = 3.2
config.num_steps = 4#3*5
config.momentum = 0.0

# ghost network related
config.optimal = False
config.random_range = 0.0
config.keep_prob = 1.0

config.beta_1=0.9
config.beta_2=0.99
config.mu_1=0.1
config.mu_2=0.01

config.attack_network_1 = "012"
config.attack_network_2 = "345"
config.attack_network_3 = "9bc"
#config.attack_network_4 = "9a"

# eval related
config.test_network = "2345016879abc"
config.eval_clean = False
config.val = False
config.GPU_ID = '3'
config.batch_size_case = 10
# misc
config.batch_size = 1
config.report_step = 100
config.overwrite = False
config.skip = False
config.img_num = -1

# data related
config.test_list_filename = '/home/ubuntu/pxq/pxq_2/ghost-network-master/ghost-network-master/data/list/test_5000.txt'
config.val_list_filename = '/home/ubuntu/pxq/pxq_2/ghost-network-master/ghost-network-master/data/list/val_list50000.txt'
config.ground_truth_file = '/home/ubuntu/pxq/pxq_2/ghost-network-master/ghost-network-master/data/valid_gt.csv'
config.img_dir = '/home/ubuntu/pxq/pxq_2/ghost-network-master/val_data/'
config.img_dir_new = '/home/ubuntu/pxq/pxq_2/ghost-network-master/ghost-network-master/data/val_data_new/'#jiade
config.checkpoint_path = "/home/ubuntu/pxq/pxq_2/ghost-network-master/ghost_checkpoint_new"
config.exp = 'I-FGSM'

parser = argparse.ArgumentParser(description='Process some integers.')
for key, value in config.items():
    if type(value) is bool:
        parser.add_argument("--" + key, action='store_' + str(not value).lower())
    else:
        parser.add_argument("--" + key, type=type(value), default=value)
args = parser.parse_args()
for key, value in args.__dict__.items():
    config[key] = value

network_pool = ["inception_v3", "inception_v4", "resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "inception_resnet_v2",
                #      0               1               2               3                4                     5
                "ens3_inception_v3", "ens4_inception_v3", "ens_inception_resnet_v2","vgg_16","vgg_19","adv_inception_v3","adv_inception_resnet_v2"
                #     6                       7                 8                       9        a         b                         c                    d
                ]

config.attack_networks = [network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index in config.attack_network]
config.test_networks = [network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index in config.test_network]
config.attack_networks_1 = [network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index in config.attack_network_1]
config.attack_networks_2 = [network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index in config.attack_network_2]
config.attack_networks_3 = [network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index in config.attack_network_3]
#config.attack_networks_4 = [network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index in config.attack_network_4]
config.result_dir = '/home/ubuntu/pxq/pxq_2/ghost-network-master/ghost-network-master/result/{:s}_{:s}_{:s}'.format(config.exp, config.attack_network,config.method)
config.result_dir_now = '/home/ubuntu/pxq/pxq_2/ghost-network-master/ghost-network-master/result/I-FGSM_0123459a_case'
if eval_mode:
    if config.eval_clean:
        if config.val:
            config.test_list_filename = config.val_list_filename
        config.result_dir = config.img_dir
    else:
        config.random_range = 0.0
        config.keep_prob = 1.0
        config.optimal = False
else:
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    else:
        assert config.overwrite or config.skip, "{:s}".format(config.result_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID

if config.skip:
    raise NotImplementedError
print(config)
