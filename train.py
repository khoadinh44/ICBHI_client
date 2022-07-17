import os
import itertools
import argparse
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.tools import to_onehot, load_df
from sklearn.metrics import confusion_matrix, accuracy_score
print ("Train import done successfully")

# input argmuments
parser = argparse.ArgumentParser(description='RespireNet: Lung Sound Classification')
parser.add_argument('--lr', default = 1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default = 16, type=int, help='bacth size')
parser.add_argument('--epochs', default = 10, type=int, help='epochs')

parser.add_argument('--save_data_dir', type=str, help='data directory: x/x/')
parser.add_argument('--data_dir', type=str, help='data directory: x/x/ICBHI_final_database')
parser.add_argument('--model_path', type=str, default = 'model.h5', help='model saving directory')

args = parser.parse_args()

################################MIXUP#####################################
def train(args):
    if os.path.exists(os.path.join(args.save_data_dir, 'test_data.pkz')):
        test_data = load_df(os.path.join(args.save_data_dir, 'test_data.pkz'))
        test_label = load_df(os.path.join(args.save_data_dir, 'test_label.pkz'))
        train_data = load_df(os.path.join(args.save_data_dir, 'train_data.pkz'))
        train_label = load_df(os.path.join(args.save_data_dir, 'train_label.pkz'))
    else:
        print('-'*10 + 'CATAGORIZE DATA' + '-'*10)
        files_name = []
        for i in args.data_dir:
            tail = i.split('.')[-1]
            head = i.split('.')[0]
            if tail == 'wav':
                files_name.append(head)

        # label (before onehot): normal, crackles, wheezes, both = 0, 1, 2, 3
        labels_data = {0: [], 1: [], 2: [], 3: []}
        for file_name in files_name:
            audio_file = file_name + '.wav'
            txt_file = file_name + '.txt'

            annotations = get_annotations(txt_file, args.data_dir)
            get_sound_samples(labels_data, annotations, audio_file, args.data_dir, sample_rate=4000)

        test_data = []
        test_label = []
        train_data = []
        train_label = []
        for name in labels_data:
            label = [name]*len(data)
            all_label = np.array([to_onehot(i) for i in label])
            all_data = labels_data[name]
            X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=42) # 80% for train, 20% for test

            if train_data == []:
                train_data = X_train
                train_label = y_train
            else:
                train_data = np.concatenate((train_data, X_train))
                train_label = np.concatenate((train_label, y_train))

            if test_data == []:
                test_data = X_test
                test_label = y_test
            else:
                test_data = np.concatenate((test_data, X_test))
                test_label = np.concatenate((test_label, y_test))

        save_df(test_data, os.path.join(args.save_data_dir, 'test_data.pkz'))
        save_df(test_label, os.path.join(args.save_data_dir, 'test_label.pkz'))
        save_df(train_data, os.path.join(args.save_data_dir, 'train_data.pkz'))
        save_df(train_label, os.path.join(args.save_data_dir, 'train_label.pkz'))
        

        
        
        
        
            


if __name__ == "__main__":
    train(args)
