import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def bilstm(input_len, n_features):
  inputs = Input(shape=(input_len, n_features))
  x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
  x = Activation('relu')(x)
  x = Bidirectional(LSTM(128, return_sequences=False))(x)
  x = Activation('relu')(x)
  outputs = Dense(12)(x)

  model = Model(inputs, outputs)
  model.summary()
  return model
