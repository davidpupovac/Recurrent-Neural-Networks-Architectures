# Python version: 3.7.3 
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf
# Scikit-learn version: 0.21.2

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
"""

                One (time step) to one (time step) sequence  

 (Multiple features, single time step -> single output features, single step)
                                
"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# =============================================================================
# =============================================================================
# =============================================================================
# Example  of regression problem


# simulate some toy data 

import numpy as np
import pandas as pd

# reproducibility
np.random.seed(10)
size = 10000
 
auto_noise = np.random.normal(scale= 2, size=size)
for j in range(2,size):
    auto_noise[j] = 0.85 * auto_noise[j-1]
    
# feature setup
var_1 = np.random.choice(a=range(1,5), size=size)[None,:]
var_2 = np.random.normal(size=size)[None,:]
var_3 = np.random.choice(a=range(2), size=size)[None,:]
var_4 = np.random.choice(a=range(2), size=size)[None,:]
var_5 = np.round(np.random.normal(loc=2.90, scale=0.5, size=size), 2)[None,:]

y = .5*var_1 + 2.2*var_2 + .75*var_3 + 1.55*var_4 + 1.55*var_4 + .125*var_5 + auto_noise

data = np.concatenate([y, var_1, var_2, var_3, var_4, var_5], axis=0)
data = np.array(data).reshape(10000, 6)

data = pd.DataFrame(data)
data.columns.values
data.columns=['y', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5']
features = ['y', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5']
data = data.reindex(columns=features)

# data['y-lag1'] = data['y'].shift(1) 
# data = data.dropna()
X = data.iloc[:, data.columns != 'y']
y = data.iloc[:, data.columns == 'y']

k_features = X.shape[1] # number of features 

# create train test splits
X_train, y_train = X.iloc[:7000], y.iloc[:7000] 
X_valid, y_valid = X.iloc[7000:9000], y.iloc[7000:9000]
X_test, y_test = X.iloc[9000:10000], y.iloc[9000:10000]

X_train = np.array(X_train).reshape(7000, 1, k_features)
X_valid = np.array(X_valid).reshape(2000, 1, k_features)
X_test = np.array(X_test).reshape(1000, 1, k_features)

# =============================================================================
# Baseline

import tensorflow as tf 
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM

# ----
# MSE - prediction based on previous step
np.square(np.subtract(y_valid.iloc[1:], y_valid.shift(1).iloc[1:])).mean() 

# MSE - Prediction based on mean 
np.square(np.subtract(y_valid, y_valid.mean())).mean() 

# ----
# define base model

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[1, k_features]), # specify dimensions of input sequence
    keras.layers.Dense(1) # predict series by series
])

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, epochs=50,  
                    validation_data=(X_valid, y_valid))  

# =============================================================================
# Simple Recurrent Neural Networks

# train 1 - simple RNN

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20,  input_shape=[1, k_features]), # input_shape=[time_steps, num_features]
    # num of parameters: (num_features + num_neurons)* num_neurons + biases
    keras.layers.Dense(1) 
    # num of parameters: num_neurons + biases
])     
    
model.compile(optimizer='adam', loss='mse')
print(model.summary())

history = model.fit(X_train, y_train, epochs=50, batch_size = 128,
                    validation_data=(X_valid, y_valid))  
# ------------
# train 1a - deep RNN

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[1, k_features]),
    # the return sequence is set to True: the output of the hidden state of 
    # each neuron is used as an input to the next LSTM layer
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1) 
    # at the last layer you can use keras.layers.SimpleRNN(1), 
    # but it's tanh activation function squashes output to -1 to 1
])     
    
model.compile(optimizer='adam', loss='mse')
print(model.summary())

history = model.fit(X_train, y_train, epochs=50,  # If unspecified, batch_size will default to 32
                    validation_data=(X_valid, y_valid))    

# =============================================================================
# Long Short-Term Memory 

# train 2 - LSTM shallow

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, k_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

import keras
optimizer = keras.optimizers.Adam(lr=0.005) #  set learning rate
model.compile(loss="mse", optimizer=optimizer)
print(model.summary())

history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# prediction 
X_test_pred = np.array(X_test).reshape(1000, 1, k_features)
y_test_pred = model.predict(X_test_pred)
y_test_pred = np.array(y_test_pred).reshape(1000, 1)
np.square(np.subtract(y_test, y_test_pred)).mean() 

# ------------
# 2a - include TensorBoard output

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, k_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

import keras
optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss="mse", optimizer=optimizer)
print(model.summary())

# get
from keras.callbacks import TensorBoard

tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
# It will generate the graph folder in your current working directory; if you want to generate it yourself:
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_valid, y_valid), verbose=1, callbacks=[tensor_board])

# ------------
# train 2b - LSTM with peephole connection

# sometimes this calls:'PeepholeLSTMCell' object has no attribute 'kernel'. Then: Restart kernel 

peephole_lstm_cells = [tf.keras.experimental.PeepholeLSTMCell(size) for size in [50, 50]] # two layers of size 50 and 50
# Create a layer composed sequentially of the peephole LSTM cells.

# peephole_lstm_cells = [tf.keras.experimental.PeepholeLSTMCell(units=50)]  

model = keras.models.Sequential([
    keras.layers.RNN(peephole_lstm_cells, input_shape=[1, k_features]),
    keras.layers.Dense(1) 
])
    
optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss="mse", optimizer=optimizer)      

print(model.summary())

history = model.fit(X_train, y_train, epochs=50, 
                    validation_data=(X_valid, y_valid))

# ------------
# train 2c - LSTM deep

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, k_features)))
model.add(LSTM(40, activation='relu',  return_sequences=True))
model.add(LSTM(30, activation='relu',  return_sequences=True))
model.add(LSTM(20, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
print(model.summary())

history = model.fit(X_train, y_train, epochs=50, 
                    validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

y_pred = model.predict(X_test)

# ------------
# train 2d - LSTM - set learning rate

import keras

optimizer = keras.optimizers.Adam(lr=0.005) # different optimizer
model.compile(loss="mse", optimizer=optimizer)

history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

y_pred = model.predict(X_test)

# ------------
# train 2e - LSTM deeper

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(1, k_features)))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
print(model.summary())

history = model.fit(X_train, y_train, epochs=50, verbose=1, 
                    validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

X_test_pred = np.array(X_test).reshape(1000, 1, k_features)
y_test_pred = model.predict(X_test_pred)
y_test_pred = np.array(y_test_pred).reshape(1000, 1)

np.square(np.subtract(y_test, y_test_pred)).mean() 

# =============================================================================
# Gated Recurrent Unit 

# train 3 - GRU

model = keras.models.Sequential([
    keras.layers.GRU(50, return_sequences=True, input_shape=[1, k_features]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(10),  
    keras.layers.Dense(1) 
])
    
optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss="mse", optimizer=optimizer)    
    
print(model.summary())

history = model.fit(X_train, y_train, epochs=50, 
                    validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# ----
# train 3a - GRU - the same with automatic batch validation split and batch size

model = keras.models.Sequential([
    keras.layers.GRU(50, return_sequences=True, input_shape=[1, k_features]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(10),  
    keras.layers.Dense(1) 
])
    
optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(loss="mse", optimizer=optimizer)   

model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=128)

# =============================================================================
# =============================================================================
# =============================================================================
# Example  of classification problem 

# simulate some toy data 

np.random.seed(10)
size = 10002 # + 2 because of two lags, you want overall size of 10,000 samples
 
# feature setup
var_1 = np.random.choice(a=range(1,5), size=size)[None,:]
var_2 = np.random.choice(a=range(2), size=size)[None,:]
var_3 = np.random.normal(scale=1.5,size=size)[None,:]

data = np.concatenate([var_1, var_2, var_3], axis=0)
data = np.array(data).reshape(size, 3)
data = pd.DataFrame(data)
data.columns=['var_1', 'var_2', 'var_3']
features = [ 'var_1', 'var_2', 'var_3']
data = data.reindex(columns=features)

def feature_lag(data, features):
    for feature in features:
        data[feature + '-lag1'] = data[feature].shift(1)
        data[feature + '-lag2'] = data[feature].shift(2)

features = ['var_1', 'var_2', 'var_3']
feature_lag(data, features)
data = data.dropna()

X=data
k_features = X.shape[1] # number of features 

y = (1.5*data['var_1'] + 2.2*data['var_2'] + 1.75*data['var_3'] + 
     .85*data['var_1-lag1'] + .55*data['var_2-lag1'] + .125*data['var_2-lag1'] +
     .55*data['var_1-lag2'] + .35*data['var_2-lag2'] + .06*data['var_2-lag2'] + np.random.normal(size=10000))

# bucketize 
from sklearn.preprocessing import KBinsDiscretizer
bucketizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform') 
y = bucketizer.fit_transform(np.array(y).reshape(-1, 1))

# create train test splits
X_train, y_train = X.iloc[:7000], y[:7000] 
X_valid, y_valid = X.iloc[7000:9000], y[7000:9000]
X_test, y_test = X.iloc[9000:10000], y[9000:10000]

X_train = np.array(X_train).reshape(7000, 1, k_features)
X_valid = np.array(X_valid).reshape(2000, 1, k_features)
X_test = np.array(X_test).reshape(1000, 1, k_features)

# ----
# train 1 - LSTM 

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(1, k_features)))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
   
history = model.fit(X_train, y_train, epochs=50, verbose=1, 
                    validation_data=(X_valid, y_valid))

# ----
# train 2 - GRU 

model = keras.models.Sequential([
    keras.layers.GRU(50, return_sequences=True, input_shape=[1, k_features]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(10),  
    keras.layers.Dense(5, activation="softmax")
])
    
optimizer = keras.optimizers.Adam(lr=0.005)

model.compile(loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])    

print(model.summary())

    
history = model.fit(X_train, y_train, epochs=50, 
                    validation_data=(X_valid, y_valid)) 
