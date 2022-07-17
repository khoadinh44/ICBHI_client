import numpy as np
import os
import io
import math
import random
import pandas as pd

import matplotlib.pyplot as plt
import librosa
import cv2

import pickle as pkl

def slice_data(start, end, raw_data, sample_rate):
    max_ind = len(raw_data) 
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]

def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    else:
        return 3
      
def get_annotations(file_name, data_dir):
  # file_name: .txt file
  f = open(os.path.join(data_dir, file_name), "r")
  f1 = [i[:-1] for i in list(f)]
  annotations = np.array([i.split('\t') for i in f1], dtype=np.float32)
  f.close()
  return annotations
      
def get_sound_samples(labels_data, annotations, file_name, data_dir, sample_rate=4000):
    # annotations: - type: list
    #              - construction: Each row of annotations is [..., ..., ..., ...] == [start, end, crackles, wheezes]
    sample_data = [file_name]
    
    # load file with specified sample rate (also converts to mono)
    data, rate = librosa.load(os.path.join(data_dir, file_name), sr=sample_rate)

    for row in annotations:
        # get annotations informations
        start = row[0]
        end = row[1]
        crackles = row[2]
        wheezes = row[3]
        
        audio_chunk = slice_data(start, end, data, rate)
        label_name = get_label(crackles, wheezes)
        labels_data[label_name].append(audio_chunk)
    return labels_data
          
def load_df(pkz_file):
    with open(pkz_file, 'rb') as f:
        df=pkl.load(f)
    return df

def save_df(df, out_file):
  with open(out_file, 'wb') as pfile:
    pkl.dump(df, pfile)
    print('{0} saved'.format(out_file))
    
def to_onehot(x, num=4):
    a = np.zeros((num, ))
    a[x] = 1
    return a.tolist()

def create_spectrograms_raw(current_window, sample_rate=4000, n_mels=224, f_min=50, f_max=4000, nfft=2048, hop=6): # increase hop -> decrease height of image
    current_window = np.array(current_window)
    S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    w, h = S.shape
    while h > w:
      hop += 2
      S = librosa.feature.melspectrogram(y=current_window, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
      w, h = S.shape
    S = librosa.power_to_db(S, ref=np.max)
    img = (S-S.min()) / (S.max() - S.min())

    if h < w:  # Padding zeros if height < width
      need = w-h
      l = need//2
      img_zer = np.zeros((w, w))
      img_zer[:, l: l+h] = S
      img = img_zer

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img
