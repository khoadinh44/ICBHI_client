import numpy as np
import os
import io
import math
import random
import pandas as pd

import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2
import cmapy

import nlpaug
import nlpaug.augmenter.audio as naa
import torch

from scipy.signal import butter, lfilter

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
          
