import numpy as np
np.random.seed(1008)
import tensorflow as tf
tf.random.set_seed(1234)
import json
from keras.callbacks import *
from keras.models import * 
from keras.layers.core import *
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, TimeDistributed ,Bidirectional,Multiply



def save_model(model,saveJ,saveW):
	json_string = model.to_json()
	with open(saveJ, 'w') as f:
		json.dump(json_string, f)
	model.save_weights(saveW)  


#Bi-LSTM without context information
def blstm(lstm_outputs,dense_outputs,vocab_size,embedding_dim,sequence_len,embedding_matrix):
    in_text = Input(shape=(sequence_len,)) 
    embedding = Embedding(vocab_size,embedding_dim,weights=[embedding_matrix],input_length=sequence_len,trainable=False)(in_text)
    bilstm = Bidirectional(LSTM(lstm_outputs,return_sequences=False, dropout=0.2))(embedding)
    # fnn = Dense(dense_outputs,activation = 'tanh')(bilstm)
    out = Dense(1, activation='sigmoid')(bilstm) 
    model = Model(inputs=in_text, outputs= out) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#Context-aware bi-LSTM
def blstm_context(lstm_outputs,dense_outputs,vocab_size,vocab_size_context,sequence_len,embedding_matrix,embedding_matrix_context):
  
  in_text = Input(shape=(sequence_len,))
  in_prev = Input(shape=(sequence_len,))
  embedding = Embedding(vocab_size,300,weights=[embedding_matrix],input_length=sequence_len,trainable=False)(in_text)
  embedding_prev = Embedding(vocab_size_context,300,weights=[embedding_matrix_context],input_length=sequence_len,trainable=False)(in_prev)
  out_text = Bidirectional(LSTM(lstm_outputs,return_sequences=False, dropout=0.2))(embedding) 
  out_prev = Bidirectional(LSTM(lstm_outputs,return_sequences=False, dropout=0.2))(embedding_prev)


  x = Concatenate(axis=-1)([out_text, out_prev])
  # fnn = Dense(dense_outputs,activation = 'tanh')(x)
  out = Dense(1, activation='sigmoid')(x) 
  model = Model(inputs=[in_text,in_prev], outputs= out)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model

