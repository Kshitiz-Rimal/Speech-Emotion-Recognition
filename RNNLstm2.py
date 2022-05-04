from keras.layers import LSTM
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout



def build_model(input_shape):
  model = Sequential([
      LSTM(128, return_sequences=False, input_shape=input_shape),
      Dense(64, activation='relu'),
      Dropout(0.4),
      Dense(32, activation='relu'),
      Dropout(0.4),
      Dense(8, activation='softmax')
  ])

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  return model