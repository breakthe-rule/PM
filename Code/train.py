'''
    Predicting next activity and next timestamp
    We are not calculating future trajectory of activities nither remaining runtime
    
'''

from __future__ import print_function, division
import numpy as np
import copy
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

# Data pat
# eventlog = "data/helpdesk.csv"
# eventlogs = ["data/BPI19.csv", "data/BPI20.csv","data/helpdesk.csv","data/bpi_12_w.csv"]
eventlogs = ["data/BPI17.csv"]
for eventlog in eventlogs:
    print("*-"*15,eventlog,"*-"*15,"\n")
    if eventlog=="data/BPI20.csv": dataset = "bpi20"
    elif eventlog=="data/helpdesk.csv": dataset="helpdesk"
    elif eventlog=="data/bpi_12_w.csv": dataset="bpi12"
    elif eventlog=="data/BPI19.csv": dataset = "bpi19"
    elif eventlog=="data/BPI17.csv": dataset = "bpi17"
    # eventlog = "data/BPI20.csv"

    lines_total,timeseqs,timeseqs2,timeseqs3,timeseqs4,numlines = read_eventlog(eventlog)

    #Standard Deviation
    divisor = np.sqrt(np.var([item for sublist in timeseqs for item in sublist]))
    divisor2 = np.sqrt(np.var([item for sublist in timeseqs2 for item in sublist]))

    # Train validation split
    elems_per_fold = int(round(numlines/3))

    # Training Data
    lines = lines_total[:2*elems_per_fold]
    lines_t = timeseqs[:2*elems_per_fold]
    lines_t2 = timeseqs2[:2*elems_per_fold]
    lines_t3 = timeseqs3[:2*elems_per_fold]
    lines_t4 = timeseqs4[:2*elems_per_fold]

    # Validation data
    fold3 = lines_total[2*elems_per_fold:]
    fold3_t = timeseqs[2*elems_per_fold:]
    fold3_t2 = timeseqs2[2*elems_per_fold:]
    fold3_t3 = timeseqs3[2*elems_per_fold:]
    fold3_t4 = timeseqs4[2*elems_per_fold:]

    #put delimiter symbol
    lines_withend = map(lambda x: x+'!',lines+fold3) 
    #find maximum line size
    maxlen = max(map(lambda x: len(x),lines_withend))

    # next lines here to get all possible characters for events and annotate them with numbers
    lines_withend = map(lambda x: x+'!',lines+fold3) 
    chars = map(lambda x: set(x),lines_withend) 
    chars = list(set().union(*chars))
    chars.sort()
    # target chars have "!" this line end symbol and chars do not.
    target_chars = copy.copy(chars)
    chars.remove("!")

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

    sentences,sentences_t,sentences_t2,sentences_t3,sentences_t4,next_chars,next_chars_t = data(lines,lines_t,lines_t2,lines_t3,lines_t4)
    sentences_val,sentences_t_val,sentences_t2_val,sentences_t3_val,sentences_t4_val,next_chars_val,next_chars_t_val = data(fold3,fold3_t,fold3_t2,fold3_t3,fold3_t4)
    print(sentences[0:10])
    print(next_chars[:10])
    print(sentences_t[:10])
    print(next_chars_t[:10])
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
    acc,mae,total_time,history = lstm_cnn(maxlen,chars,target_chars,CX,CX_val,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char,divisor,dataset)
    print(acc,mae,total_time)
    val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)
    print()

    # print("ONLY LSTM")
    # acc,mae,total_time,history = lstm(maxlen,chars,target_chars,CX,CX_val,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char,divisor,dataset)
    # print(acc,mae,total_time)
    # val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)
    # print()

    # print("ONLY CNN")
    # acc,mae,total_time,history = cnn(maxlen,chars,target_chars,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char,divisor,dataset)
    # print(acc,mae,total_time)
    # val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)
    # print()

    # print("Random forest")
    # acc,mae,total_time = randomforest(X, y_a, y_t, X_val, y_a_val, y_t_val,divisor,dataset)
    # print(acc,mae,total_time)
    # val_accuracy.append(acc); val_mean_average_error.append(mae); train_time.append(total_time)
    # print()

    print("#--*"*30)
    print("val_accuracy:",val_accuracy)
    print("val_mean_average_error:",val_mean_average_error)
    print("train_time:",train_time)
    print()