#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')


parser.add_argument('--dataset', default="carom")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)


parser.add_argument('--num_f_maps', default='64', type=int)

#asformer
parser.add_argument('--num_layers', default='10',type=int)
parser.add_argument('--r1', default='2',type=int)
parser.add_argument('--r2', default='2',type=int)
parser.add_argument('--channel_masking_rate', default='0.3',type=float)

# Need input
parser.add_argument(
    "--arch",
    default="asformer",
    choices=["asformer", "MSTCN"], 
    type=str,
)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)

# david
parser.add_argument('--bce_pos_weight', type=float, default=1, help='')
#



args = parser.parse_args()

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2
elif args.dataset == "carom":
    sample_rate = 1
# print(args.dataset)
root = r'/home/oort/MS-TCN2/'
vid_list_file = root + "data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = root + "data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
# david
vid_list_file_val = root + "data/"+args.dataset+"/splits/val.split"+args.split+".bundle"
#
features_path = root + "data/"+args.dataset+"/features/"
gt_path = root + "data/"+args.dataset+"/groundTruth/"

mapping_file = root + "data/"+args.dataset+"/mapping.txt"

model_dir ="./models/"+args.dataset+"/split_"+args.split
results_dir ="./results/"+args.dataset+"/split_"+args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
trainer = Trainer(args, num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split, device=device)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    # david
    # validate
    batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_val.read_data(vid_list_file_val)
    #
    print("Training Dataloaded")
    trainer.train(model_dir, batch_gen, batch_gen_val, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    #trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs,
                    actions_dict, device, sample_rate, gt_path, mapping_file)
