import os
import itertools
import argparse
import random
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler


import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load external modules
from utils.tools import *
from utils.image_dataloader import *
#from nets.network_hybrid import *
from sklearn.metrics import confusion_matrix, accuracy_score
print ("Train import done successfully")

# input argmuments
parser = argparse.ArgumentParser(description='RespireNet: Lung Sound Classification')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0005,help='weight decay value')
parser.add_argument('--gpu_ids', default=[0,1], help='a list of gpus')
parser.add_argument('--num_worker', default=4, type=int, help='numbers of worker')
parser.add_argument('--batch_size', default=4, type=int, help='bacth size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--start_epochs', default=0, type=int, help='start epochs')

parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--folds_file', type=str, help='folds text file')
parser.add_argument('--test_fold', default=4, type=int, help='Test Fold ID')
parser.add_argument('--stetho_id', default=-1, type=int, help='Stethoscope device id')
parser.add_argument('--aug_scale', default=None, type=float, help='Augmentation multiplier')
parser.add_argument('--model_path',type=str, help='model saving directory')
parser.add_argument('--checkpoint', default=None, type=str, help='load checkpoint')

args = parser.parse_args()

################################MIXUP#####################################
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

##############################################################################
def get_score(hits, counts, pflag=False):
    se = (hits[1] + hits[2] + hits[3]) / (counts[1] + counts[2] + counts[3])
    sp = hits[0] / counts[0]
    sc = (se+sp) / 2.0

    if pflag:
        print("*************Metrics******************")
        print("Se: {}, Sp: {}, Score: {}".format(se, sp, sc))
        print("Normal: {}, Crackle: {}, Wheeze: {}, Both: {}".format(hits[0]/counts[0], hits[1]/counts[1], 
            hits[2]/counts[2], hits[3]/counts[3]))
    

mean, std = get_mean_and_std(image_loader(self.args.data_dir, self.args.folds_file, self.args.test_fold, True, "Params_json", Compose([ToTensor()]), stetho_id=self.args.stetho_id))
print("MEAN",  mean, "STD", std)

input_transform = Compose([ToTensor(), Normalize(mean, std)])
train_dataset = image_loader(self.args.data_dir, self.args.folds_file, self.args.test_fold, True, "params_json", self.input_transform, stetho_id=self.args.stetho_id, aug_scale=self.args.aug_scale)
test_dataset = image_loader(self.args.data_dir, self.args.folds_file, self.args.test_fold,  False, "params_json", self.input_transform, stetho_id=self.args.stetho_id)
test_ids = np.array(test_dataset.identifiers)
test_paths = test_dataset.filenames_with_labels



# if __name__ == "__main__":
#     trainer = Trainer()
#     trainer.train()
