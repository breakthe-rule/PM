from __future__ import print_function, division
import keras
from keras.utils import get_file #Change
from collections import Counter
import unicodecsv
import numpy as np
import random
import sys
import os
import copy
import time
import random
from datetime import datetime
from math import log
from sklearn.metrics import accuracy_score
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import matplotlib.pyplot as plt

# Importing functions
from read_eventlog import read_eventlog
from data import data
from vectorize import vectorize
from vectorize_senlen import vextorize_senlen
from senlen import senlen
from lstm_cnn import lstm_cnn
from lstm import lstm
from cnn import cnn
from randomforest import randomforest

# Data path
# eventlog = "data/helpdesk.csv"
eventlog = "data/bpi_12_w.csv"

lines_total,timeseqs,timeseqs2,timeseqs3,timeseqs4,numlines = read_eventlog(eventlog)

#Standard Deviation
divisor = np.sqrt(np.var([item for sublist in timeseqs for item in sublist]))
divisor2 = np.sqrt(np.var([item for sublist in timeseqs2 for item in sublist]))

# Train validation split
elems_per_fold = int(round(numlines/3))

lines = lines_total[:2*elems_per_fold]
lines_t = timeseqs[:2*elems_per_fold]
lines_t2 = timeseqs2[:2*elems_per_fold]
lines_t3 = timeseqs3[:2*elems_per_fold]
lines_t4 = timeseqs4[:2*elems_per_fold]

fold3 = lines_total[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
fold3_t4 = timeseqs4[2*elems_per_fold:]

# Some useful variables
lines_withend = map(lambda x: x+'!',lines+fold3) #put delimiter symbol 
maxlen = max(map(lambda x: len(x),lines_withend)) #find maximum line size
# next lines here to get all possible characters for events and annotate them with numbers
lines_withend = map(lambda x: x+'!',lines+fold3) 
chars = map(lambda x: set(x),lines_withend) 
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove("!")
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

sentences,sentences_t,sentences_t2,sentences_t3,sentences_t4,next_chars,next_chars_t = data(lines,lines_t,lines_t2,lines_t3,lines_t4)
sentences_val,sentences_t_val,sentences_t2_val,sentences_t3_val,sentences_t4_val,next_chars_val,next_chars_t_val = data(fold3,fold3_t,fold3_t2,fold3_t3,fold3_t4)

val_accuracy = []
val_mean_average_error = []
train_time = []

'''
Sentence lenght
'''
#LSTM
# X,Cy_a,y_t = vextorize_senlen(char_indices,divisor,divisor2,next_chars_t,next_chars,chars,sentences,maxlen,target_chars,target_char_indices
#               ,sentences_t,sentences_t2,sentences_t3,sentences_t4)

# X_val,y_a_val,y_t_val = vextorize_senlen(char_indices,divisor,divisor2,next_chars_t_val,next_chars_val,chars,sentences_val,maxlen,target_chars,target_char_indices
#               ,sentences_t_val,sentences_t2_val,sentences_t3_val,sentences_t4_val)

# #CNN
# CX,y_a,Cy_t = vextorize_senlen(char_indices,divisor,divisor2,next_chars_t,next_chars,chars,sentences,maxlen,target_chars,target_char_indices)

# CX_val,y_a_val,y_t_val = vextorize_senlen(char_indices,divisor,divisor2,next_chars_t_val,next_chars_val,chars,sentences_val,maxlen,target_chars,target_char_indices)

# print("LSTM+CNN with sentence lenght")
# acc,mae,total_time,history = senlen(maxlen,chars,target_chars,CX,CX_val,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char)
# print(acc,mae,total_time)
# val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)


'''
NO sentence lenght
'''
#LSTM
X,Cy_a,y_t = vectorize(char_indices,divisor,divisor2,next_chars_t,next_chars,chars,sentences,maxlen,target_chars,target_char_indices
              ,sentences_t,sentences_t2,sentences_t3,sentences_t4)

X_val,y_a_val,y_t_val = vectorize(char_indices,divisor,divisor2,next_chars_t_val,next_chars_val,chars,sentences_val,maxlen,target_chars,target_char_indices
              ,sentences_t_val,sentences_t2_val,sentences_t3_val,sentences_t4_val)

#CNN
CX,y_a,Cy_t = vectorize(char_indices,divisor,divisor2,next_chars_t,next_chars,chars,sentences,maxlen,target_chars,target_char_indices)

CX_val,y_a_val,y_t_val = vectorize(char_indices,divisor,divisor2,next_chars_t_val,next_chars_val,chars,sentences_val,maxlen,target_chars,target_char_indices)

print("LSTM+CNN")
acc,mae,total_time,history = lstm_cnn(maxlen,chars,target_chars,CX,CX_val,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char,divisor)
print(acc,mae,total_time)
val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)

print("ONLY LSTM")
acc,mae,total_time,history = lstm(maxlen,chars,target_chars,CX,CX_val,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char,divisor)
print(acc,mae,total_time)
val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)

print("ONLY CNN")
acc,mae,total_time,history = cnn(maxlen,chars,target_chars,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char,divisor)
print(acc,mae,total_time)
val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)

print("Random forest")
acc,mae,total_time = randomforest(X, y_a, y_t, X_val, y_a_val, y_t_val,divisor)
print(acc,mae,total_time)
val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)

print("val_accuracy:",val_accuracy)
print("val_mean_average_error:",val_mean_average_error)
print("train_time:",train_time)