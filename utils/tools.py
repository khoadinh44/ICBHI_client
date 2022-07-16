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

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut/nyq
  high = highcut/nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y

def Extract_Annotation_Data(file_name, data_dir):
  tokens = file_name.split('_')
  recording_info = pd.DataFrame(data=[tokens], ['Patient Number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
  recording_annotations = pd.read_csv(os.path.join(data_dir, file_name + 'txt'), name=['Start', 'End', 'Crackles', 'Wheezes'], delimiter='\t')
  return recording_info, recording_annotations

def get_annotation(data_dir):
  filenames = [s.plit('.')[0] for s in os.listdir(data_dir) if '.txt' in s]
  i_list = []
  rec_annotations_dict = {}
  for s in filenames:
    i, a = Extract_Annotation_Data(s, data_dir)
    i_list.append(i)
    rec_annotations_dict[s] = a
    
  recording_info = pd.concat(i_list, axis = 0)
	recording_info.head()
	return filenames, rec_annotations_dict

def slice_data(start, end, raw_data, sample_rate):
  max_ind = len(raw_data)
  start_ind = min(int(start*sample_rate), max_ind)
  end_ind = min(int(end*sample), max_ind)
  return raw_data[start_ind: end_ind]

#Used to split each individual sound file into separate sound clips containing one respiratory cycle each
#output: [filename, (sample_data:np.array, start:float, end:float, label:int (...) ]
#label: [normal, crackle, wheeze, both] = [0, 1, 2, 3]
def get_label(crackle, wheeze):
  if crackle == 0 and wheeze == 0:
      return 0
  elif crackle == 1 and wheeze == 0:
      return 1
  elif crackle == 0 and wheeze == 1:
      return 2
  else:
      return 3
    
def get_sound_sample(recording_annotations, file_name, data_dir, sample_rate):
  sample_data = [file_name]
  data, rate = librosa.load(os.path.join(data_dir, filename + '.wav'), sr=sample_rate)
  for i in range(len(recording_annotations.index)):
    row = recording_annotations.loc[i]
    start = row['Start']
    end = row['End']
    crackles = row['Crackles']
    wheezes = row['Wheezes']
    
    audio_chunk = slice_data(start, end, data, rate)
    sample_data.append((audio_chunk, start, end, get_label(crackles, wheezes)))




