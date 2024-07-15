import keras
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout #Change,
from keras.layers import LSTM, GRU, SimpleRNN #Change
from keras.layers import Input
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization #Change
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Concatenate, Input,AveragePooling1D)
from tensorflow.keras.regularizers import L1, L2
import time
import numpy as np

def lstm(maxlen,chars,target_chars,CX,CX_val,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char,divisor,dataset):
  # Define input layers
  main_input = Input(shape=(maxlen, len(chars)+5), name='main_input')

  ''' HelpDesk '''
  if dataset == "helpdesk":
    l1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
    b1 = BatchNormalization()(l1)
    
    #branching
    l2_1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)
    l2_2 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
    b2_2 = BatchNormalization()(l2_2)
    
    #output
    act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)
    
    # Create model
    model = Model(inputs=[main_input], outputs=[act_output, time_output])

    # Compile model
    opt = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.998, epsilon=1e-07)
    model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
    
    early_stopping = EarlyStopping(monitor='val_loss',patience=42, mode="min")
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
    
  
  ''' BPI_12_W'''
  if dataset == "bpi12":
    l1 = LSTM(32, kernel_initializer='glorot_uniform', return_sequences=True)(main_input) # the shared layer
    b1 = BatchNormalization()(l1)
    l2 = LSTM(16, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) 
    b2 = BatchNormalization()(l2)
    
    #branching
    # l2_1 = LSTM(16, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
    # b2_1 = BatchNormalization()(l2_1)
    # l2_2 = LSTM(16, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
    # b2_2 = BatchNormalization()(l2_2)

    #output
    b2 = Dense(32,activation = "relu")(b2)
    act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2)
    b2 = Dense(16,activation = "relu")(b2)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2)
    
    # Create model
    model = Model(inputs=[main_input], outputs=[act_output, time_output])

    # Compile model
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.998, epsilon=1e-07)
    model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
    
    early_stopping = EarlyStopping(monitor='val_loss',patience=10, mode="min")
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0)
  
  ''' BPI_20 '''
  if dataset=="bpi20":
    l1 = LSTM(64, kernel_initializer='glorot_uniform', return_sequences=True)(main_input) # the shared layer
    b1 = BatchNormalization()(l1)
    
    #branching
    l2_1 = LSTM(64, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)
    l2_2 = LSTM(40, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
    b2_2 = BatchNormalization()(l2_2)
    
    #output
    act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)
    
    # Create model
    model = Model(inputs=[main_input], outputs=[act_output, time_output])

    # Compile model
    opt = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.998, epsilon=1e-07)
    model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
    
    early_stopping = EarlyStopping(monitor='val_loss',patience=42, mode="min")
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
    
  '''Constant'''
  train_data = (X)  # Assuming lstm_input is derived from  X
  train_labels = {'act_output': y_a, 'time_output': y_t}

  val_data = (X_val)
  val_labels = {'act_output': y_a_val, 'time_output': y_t_val}
  
  start = time.time()
  
  if dataset=="helpdesk":
    history = model.fit(train_data, train_labels, validation_data=(val_data,val_labels), verbose=0, 
                          callbacks=[ lr_reducer], epochs=30)
  else:
    history = model.fit(train_data, train_labels, validation_data=(val_data,val_labels), 
                        callbacks=[ lr_reducer,early_stopping], epochs=100,verbose=0)

  end = time.time()
  pred = model.predict(X_val)
  pred_symbol = [target_indices_char[np.argmax(i)] for i in pred[0]]
  y = [target_indices_char[np.argmax(i)] for i in y_a_val]
  accuracy = accuracy_score(y, pred_symbol)
  # print(f"Accuracy: {accuracy}")

  pred_time = [i[0] for i in pred[1]]
  mae = (np.mean(np.abs(pred_time - y_t_val)))*divisor/86400
  # print("MAE:", mae)
  return accuracy,mae,end-start,history

# Example usage
# model = cnn_lstm(maxlen=30, num_chars=20, target_chars=['a', 'b', 'c'])
# model.summary()
