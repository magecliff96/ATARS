import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter  # Step 1: Import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', choices=['train', 'predict'])
parser.add_argument(
    "--arch",
    default="asformer",
    choices=["asformer", "UVAST"], 
    type=str,
)
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')
parser.add_argument('--load_model', default=True)
parser.add_argument('--pretrain', default=r"/home/magecliff/Traffic_Recognition/ASFormer-test/models/weights/15")

##UVAST settings
parser.add_argument('--use_pe_tgt', default=True, action='store_true', help='use positional encoding for target in the transcript decoder')
parser.add_argument('--use_pe_src', default=False, action='store_true', help='use positional encoding for source in the transcript decoder')
parser.add_argument('--use_alignment_dec', default=False, action='store_true', help='use alignment decoder for duration prediction (second stage)')
parser.add_argument('--alignment_decoder_model', default='uvast_decoder', type=str, choices=['uvast_decoder', 'pytorch_decoder'], help='select alignment decoder model')
parser.add_argument('--encoder_model', default='asformer_advanced', type=str, choices=['asformer_advanced', 'asformer_org_enc', 'asformer_org_enc_dec'], help='select encoder model')
parser.add_argument('--add_tgt_pe_dec_dur', default=0, type=float, help='set to 1 to add pe in the alignment decoder')
parser.add_argument('--split_segments', default=True, action='store_true', help='splitting segments')
parser.add_argument('--n_head_dec', default=1, type=int, help='the number of heads of the transcript decoder (first stage)')
parser.add_argument('--split_segments_max_dur', default='0.1', type=float, help='max duration in split_segments; for details see the paper')


parser.add_argument('--temperature', default=0.001, type=float, help='the temperature in the cross attention loss')
parser.add_argument('--do_framewise_loss', default=True, action='store_true', help='use frame wise loss after encoder')
parser.add_argument('--do_framewise_loss_g', default=True, action='store_true', help='use group wise frame wise loss')
parser.add_argument('--do_segwise_loss', default=True, action='store_true', help='use segment wise ce loss')
parser.add_argument('--do_segwise_loss_g', default=True, action='store_true', help='use group segment wise ce loss')
parser.add_argument('--do_crossattention_action_loss_nll', default=True, action='store_true', help='use cross attention loss for first stage')
parser.add_argument('--do_crossattention_dur_loss_ce', default=False, action='store_true', help='use cross attention loss for second stage')
parser.add_argument('--framewise_loss_g_apply_logsoftmax', default=False, action='store_true', help='type of the normalization for group wise CE') 
parser.add_argument('--framewise_loss_g_apply_nothing', default=True, action='store_true', help='type of the normalization for group wise CE-this is normal averaging')
parser.add_argument('--segwise_loss_g_apply_logsoftmax', default=True, action='store_true', help='type of the normalization for group wise CE')  
parser.add_argument('--segwise_loss_g_apply_nothing', default=False, action='store_true', help='type of the normalization for group wise CE-this is normal averaging')
parser.add_argument('--extra_loss', default=True, action='store_true', help='type of the normalization for group wise CE-this is normal averaging') 


args = parser.parse_args()
args.device = device

num_epochs = 50

lr = 1e-6 #1e-3
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1

channel_mask_rate = 0.3

# use the full temporal resolution @ 15fps
sample_rate = 1


vid_list_file = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/bundles/train.split.bundle"
vid_list_file_val = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/bundles/val.split.bundle"
vid_list_file_tst = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/bundles/test.split.bundle"
features_path = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/features/imgnet/"
gt_path = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/labels/"
 
mapping_file = r"/home/magecliff/Traffic_Recognition/Carom_TempSeg/mapping.txt"

# vid_list_file = r"/home/magecliff/Traffic_Recognition/mini_tempseg/mini_bundles/train.split.bundle"
# vid_list_file_val = r"/home/magecliff/Traffic_Recognition/mini_tempseg/mini_bundles/val.split.bundle"
# vid_list_file_tst = r"/home/magecliff/Traffic_Recognition/mini_tempseg/mini_bundles/test.split.bundle"
# features_path = r"/home/magecliff/Traffic_Recognition/mini_tempseg/mini_features/"
# gt_path = r"/home/magecliff/Traffic_Recognition/mini_tempseg/mini_labels/"
 
# mapping_file = r"/home/magecliff/Traffic_Recognition/mini_tempseg/mapping.txt"
 
model_dir = "./{}/".format(args.model_dir)+args.arch
results_dir = "./{}/".format(args.result_dir)+args.arch
writer = SummaryWriter(log_dir=results_dir)

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
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate, args)
if args.load_model == True:
    trainer.load_model(args.pretrain + ".model")


if args.action == "train":
    # Initialize the training data generator and preload data
    batch_gen = BatchGenerator(args, num_classes, actions_dict, gt_path, features_path, sample_rate, "train")
    batch_gen.read_data(vid_list_file)  # Preload training data

    # Initialize the validation data generator and preload data
    batch_gen_val = BatchGenerator(args, num_classes, actions_dict, gt_path, features_path, sample_rate, 'val')
    batch_gen_val.read_data(vid_list_file_val)  # Preload validation data

    batch_gen.reset()  # Shuffle data for the first epoch
    batch_gen_val.reset()  # Shuffle validation data (if required)

    # Start training the model
    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, args, batch_gen_val, writer=writer)

if args.action == "predict":
    # Initialize the testing data generator and preload data
    batch_gen_tst = BatchGenerator(args, num_classes, actions_dict, gt_path, features_path, sample_rate, 'val')
    batch_gen_tst.read_data(vid_list_file_tst)  # Preload test data

    # Start prediction
    trainer.predict(args, model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

writer.close()

