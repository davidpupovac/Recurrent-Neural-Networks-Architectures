# Python version: 3.7.3 
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
"""

               Many (time step) to one (time step) sequence 2  
                
 (Multiple features, many time steps -> multiple output features, single step)
 
                                
"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# =============================================================================
# =============================================================================
# =============================================================================
# Example  of regression problem


# simulate some toy data 2

import pandas as pd
import numpy as np

num_samples = 1000
time_steps = 20
k_features=6

def data_gen (num_samples,time_steps, k_features):
    X = np.empty([num_samples, time_steps, k_features])
    y = np.empty([num_samples, k_features])
    for i in range(0,num_samples):
        for j in range(0,k_features):
            X[i, :, j]= np.random.normal(size=time_steps)
            y[i, j]= sum(X[i, :, j]) 
    return X, y

X,y=data_gen(num_samples, time_steps, k_features)

train_size = int(np.around([len(X)*0.7]))
valid_size = int(np.around([len(X)*0.2]))

X_train, y_train = X[:train_size,:,:], y[:train_size] 
X_valid, y_valid = X[train_size: train_size+valid_size,:,:], y[train_size:train_size+valid_size]
X_test, y_test = X[train_size+valid_size:len(X)], y[train_size+valid_size:len(X)]

X_train = np.array(X_train).reshape(train_size, time_steps, k_features)
X_valid = np.array(X_valid).reshape(valid_size, time_steps, k_features)

# =============================================================================
# train 1 - LSTM  

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, k_features))) # input_shape= 100 time steps, 9 features
model.add(Dense(k_features)) # output is one step
model.compile(optimizer='adam', loss='mse')

print(model.summary())

history = model.fit(X_train, y_train, epochs=50, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)
 
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# prediction 
y_test_pred = model.predict(X_test)
y_test_pred.shape # for each set you get k_features values
np.square(np.subtract(y_test, y_test_pred)).mean() 

# --------
# train 1a - LSTM deeper (with BatchNormalization which does not really work well here)
   
from keras.layers import BatchNormalization

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, k_features)))
# model.add(BatchNormalization()) 
model.add(LSTM(10, activation='relu'))
model.add(Dense(k_features))
model.compile(optimizer='adam', loss='mse')
print(model.summary())

history = model.fit(X_train, y_train, epochs=50, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# =============================================================================
# train 2  - Bidirectional LSTM

from keras.layers import Bidirectional

model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(time_steps, k_features)))
model.add(Dense(k_features))
model.compile(optimizer='adam', loss='mse')

print(model.summary())

history = model.fit(X_train, y_train, epochs=50, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# predict
test_output = model.predict(X_test, verbose=0)

# =============================================================================
# train 3 - GRU 

import keras # for keras.layers.GRU to work, you can specify it in other ways

model = Sequential()
model.add(keras.layers.GRU(10, activation='relu', input_shape=(time_steps, k_features))) 
model.add(Dense(k_features)) # output is one step
model.compile(optimizer='adam', loss='mse')

print(model.summary())

history = model.fit(X_train, y_train, epochs=50, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# =============================================================================
# train 4 - Conv1D-LTSM  

from keras.layers import Input, Dense, Conv1D, Flatten
from keras.models import Model

input_layer = Input(shape=(time_steps, k_features))
conv1 = Conv1D(filters=32,
               kernel_size=9, padding="same", # both padding same "same" and "valid" works
               strides=1,
               activation='relu')(input_layer)
lstm1 = LSTM(10, return_sequences=False)(conv1)
output_layer = Dense(k_features)(lstm1)
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=10, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# =============================================================================
# train 5  WaveNet-like 

import keras

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, k_features]))
for rate in (1, 2, 4, 8): #  using growing dilation rates: 1, 2, 4, 8
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="valid",
                                  activation="relu", dilation_rate=rate))
model.add(keras.layers. LSTM(16)) 
model.add(keras.layers.Dense(k_features))

print(model.summary()) # 

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=50, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])