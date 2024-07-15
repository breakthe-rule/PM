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

def lstm_cnn(maxlen,chars,target_chars,CX,CX_val,X,X_val,y_a,y_a_val,y_t,y_t_val,target_indices_char,divisor,dataset):
  # Define input layers
  cnn_input = Input(shape=(maxlen, len(chars) + 1))
  lstm_input = Input(shape=(maxlen, len(chars) + 5))

  ''' HelpDesk'''
  
  if dataset=="helpdesk":
    # CNN branch
    cnn_layer1 = Conv1D(filters=32, kernel_size=2, activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=L2(l2=0.01))(cnn_input)
    cnn_layer1 = BatchNormalization()(cnn_layer1)
    cnn_pool1 = AveragePooling1D(pool_size=2)(cnn_layer1)
    cnn_flatten = Flatten()(cnn_pool1)

    # LSTM branch
    lstm_layer = LSTM(80,  kernel_initializer='glorot_uniform',return_sequences=False,activity_regularizer=L2(l2=0.01))(lstm_input)
    lstm_layer = BatchNormalization()(lstm_layer)

    # Feature fusion
    merged = Concatenate()([cnn_flatten, lstm_layer])

    # Activity prediction branch
    dense_act1 = Dense(64, activation='relu',activity_regularizer=L2(l2=0.001), kernel_initializer='glorot_uniform')(merged)
    act_output = Dense(len(target_chars), activation='softmax', name='act_output',kernel_initializer='glorot_uniform')(dense_act1)

    # Time prediction branch
    dense_time1 = Dense(60, activation='elu',activity_regularizer=L2(l2=0.01))(merged)
    time_output = Dense(1, activation='relu', name='time_output')(dense_time1)

    # Create model
    model = tf.keras.Model(inputs=[cnn_input, lstm_input], outputs=[act_output, time_output])

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.998, epsilon=1e-07)
    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=optimizer)
    
    early_stopping = EarlyStopping(monitor='val_act_output_loss', patience=10,verbose=1)
    # learning_rate_scheduler = ReduceLROnPlateau(monitor='val_act_output_loss', factor=0.5, patience=3,min_lr=0.00001,verbose=1)
    learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,min_lr=0.00001,verbose=0)
    
  ''' BPI_12_W '''
  if dataset=="bpi12":
    # CNN branch
    cnn_layer1 = Conv1D(filters=32, kernel_size=2, activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=L2(l2=0.01))(cnn_input)  
    cnn_layer1 = BatchNormalization()(cnn_layer1)
    cnn_pool1 = AveragePooling1D(pool_size=2)(cnn_layer1)
    cnn_flatten = Flatten()(cnn_pool1)

    # LSTM branch
    lstm_layer = LSTM(16,  kernel_initializer='glorot_uniform',return_sequences=False)(lstm_input)
    lstm_layer = BatchNormalization()(lstm_layer)

    # Feature fusion
    merged = Concatenate()([cnn_flatten, lstm_layer])

    # Activity prediction branch
    dense_act1 = Dense(24, activation='relu',activity_regularizer=L2(l2=0.001), kernel_initializer='glorot_uniform')(merged)
    act_output = Dense(len(target_chars), activation='softmax', name='act_output',kernel_initializer='glorot_uniform')(dense_act1)

    # Time prediction branch
    dense_time1 = Dense(24, activation='elu',activity_regularizer=L2(l2=0.001))(merged)
    time_output = Dense(1, activation='relu', name='time_output')(dense_time1)

    # Create model
    model = tf.keras.Model(inputs=[cnn_input, lstm_input], outputs=[act_output, time_output])

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.998, epsilon=1e-07)
    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=optimizer)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,verbose=1,mode="min",restore_best_weights=True)
    learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,verbose=0)

  ''' BPI 20 '''
  if dataset=="bpi20":
    # CNN branch
    cnn_layer1 = Conv1D(filters=32, kernel_size=2, activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=L2(l2=0.01))(cnn_input)  
    cnn_layer1 = BatchNormalization()(cnn_layer1)
    cnn_layer1 = Conv1D(filters=32, kernel_size=2, activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=L2(l2=0.01))(cnn_layer1)  
    cnn_layer1 = Dropout(rate=0.3)(cnn_layer1) 
    cnn_layer1 = BatchNormalization()(cnn_layer1)
    cnn_pool1 = AveragePooling1D(pool_size=2)(cnn_layer1)
    cnn_flatten = Flatten()(cnn_pool1)

    # LSTM branch
    lstm_layer = LSTM(64,  kernel_initializer='glorot_uniform',return_sequences=False)(lstm_input)
    lstm_layer = BatchNormalization()(lstm_layer)

    # Feature fusion
    merged = Concatenate()([cnn_flatten, lstm_layer])

    # Activity prediction branch
    dense_act1 = Dense(16, activation='relu',activity_regularizer=L2(l2=0.001), kernel_initializer='glorot_uniform')(merged)
    act_output = Dense(len(target_chars), activation='softmax', name='act_output',kernel_initializer='glorot_uniform')(dense_act1)

    # Time prediction branch
    dense_time1 = Dense(32, activation='elu',activity_regularizer=L2(l2=0.001))(merged)
    time_output = Dense(1, activation='relu', name='time_output')(dense_time1)

    # Create model
    model = tf.keras.Model(inputs=[cnn_input, lstm_input], outputs=[act_output, time_output])

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.998, epsilon=1e-07)
    model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=optimizer)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,verbose=1,mode="min",restore_best_weights=True)
    learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,verbose=0)

  '''Constant'''
  train_data = (CX, X)  # Assuming lstm_input is derived from X
  train_labels = {'act_output': y_a, 'time_output': y_t}

  val_data = (CX_val, X_val)
  val_labels = {'act_output': y_a_val, 'time_output': y_t_val}
  
  start = time.time()
  if dataset=="helpdesk":
    history = model.fit(train_data, train_labels,
                        epochs=25,
                        callbacks=[ learning_rate_scheduler],
                        validation_data=(val_data,val_labels),
                        verbose=0)
  else:
    history = model.fit(train_data, train_labels,
                      epochs=100,
                      callbacks=[ learning_rate_scheduler,early_stopping],
                      validation_data=(val_data,val_labels),
                      verbose=0)
  
  end = time.time()
  pred = model.predict([CX_val,X_val])
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
