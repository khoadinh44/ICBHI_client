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

# from utils.tools import 
from sklearn.metrics import confusion_matrix, accuracy_score
print ("Train import done successfully")

# input argmuments
parser = argparse.ArgumentParser(description='RespireNet: Lung Sound Classification')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=16, type=int, help='bacth size')
parser.add_argument('--epochs', default=10, type=int, help='epochs')

parser.add_argument('--data_dir', type=str, help='data directory')
parser.add_argument('--model_path',type=str, help='model saving directory')

args = parser.parse_args()

################################MIXUP#####################################
def train(args):
    print('-'*10 + 'CATAGORIZE DATA' + '-'*10)
    files_name = []
    for i in args.data_dir:
        tail = i.split('.')[-1]
        head = i.split('.')[0]
        if tail == 'wav':
            files_name.append(head)
            
    # label: normal, crackles, wheezes, both = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    for file_name in files_name:
        audio_file = file_name + '.wav'
    
            


if __name__ == "__main__":
    train(args)
