# Python version: 3.7.3 
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
"""

                Many (time step) to one (time step) sequence  
                
 (Multiple features, many time steps -> single output feature, single step)
 
                                
"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# =============================================================================
# =============================================================================
# =============================================================================
# Example  of regression problem

 
# Simulate some toy data 
        
import pandas as pd
import numpy as np

np.random.seed(10)
size = 1002 # + 2 because of two lags, you want overall size of 1,000 samples
 
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
     .85*data['var_1-lag1'] + .55*data['var_2-lag1'] + .625*data['var_2-lag1'] +
     .55*data['var_1-lag2'] + .35*data['var_2-lag2'] + .125*data['var_2-lag2'] + np.random.normal(size=1000))

# -----------------------------------------------------------------------------
# Create windows 

from itertools import islice

def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

time_steps= 100 # arbitrary number of time steps selected
num_wind = X.shape[0] + 1 - time_steps

X_stack = np.empty([num_wind, time_steps])
X_stack.shape
# empty stacked variable (zeros) - consisting of  1000-100+1=901 windows of 100 time steps

for column in X:
   x = list(window(X[column], time_steps))
   X_stack=np.dstack([X_stack,x])

X_stack = X_stack[:, :, 1:k_features+1] # remove first (empty) variable
X_stack.shape

# ----
# define y 

y_stack = list(window(y, time_steps))
y_stack=np.array(y_stack)
y_stack = y_stack[:,time_steps-1]  # select the last step

# ----
# train test split 
train_size = int(np.around([len(X_stack)*0.7]))
valid_size = int(np.around([len(X_stack)*0.2]))

X_train, y_train = X_stack[:train_size,:,:], y_stack[:train_size] 
X_valid, y_valid = X_stack[train_size: train_size+valid_size,:,:], y_stack[train_size:train_size+valid_size]
X_test, y_test = X_stack[train_size+valid_size:len(X_stack)], y_stack[train_size+valid_size:len(X_stack)]

X_train = np.array(X_train).reshape(train_size, time_steps, k_features)
X_valid = np.array(X_valid).reshape(valid_size, time_steps, k_features)

# =============================================================================
# Baseline

y_valid_ = pd.DataFrame(y_valid) # temporary pandas feature

# ----
# MSE - prediction based on previous step
np.square(np.subtract(y_valid_.iloc[1:], y_valid_.shift(1).iloc[1:])).mean() 

# MSE - Prediction based on mean 
np.square(np.subtract(y_valid_, y_valid_.mean())).mean() 

del(y_valid_) # delete temporary pandas feature

# =============================================================================
# train 1 - LSTM - Long Short-Term Memory 

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(time_steps, k_features))) # input_shape= 100 time steps, 9 feature
model.add(Dense(1)) # output is one step
model.compile(optimizer='adam', loss='mse')

print(model.summary())

history = model.fit(X_train, y_train, epochs=10, verbose=1,
                    validation_data=(X_valid, y_valid))

# =============================================================================
# train 2 - GRU - Gated Recurrent Unit 

import keras # for keras.layers.GRU to work, you can specify it in other ways

model = Sequential()
model.add(keras.layers.GRU(10, activation='relu', input_shape=(time_steps, k_features))) 
model.add(Dense(1)) # output is one step
model.compile(optimizer='adam', loss='mse')

print(model.summary())

history = model.fit(X_train, y_train, epochs=10, verbose=1,
                    validation_data=(X_valid, y_valid))

# =============================================================================
# train 3a - Conv1D-LTSM works - Convolutional layer to LSTM

from keras.layers import Input, Dense, Conv1D, Flatten
from keras.models import Model

input_layer = Input(shape=(time_steps, k_features))
conv1 = Conv1D(filters=32,
               kernel_size=9, padding="same", # both padding "same" and "valid" works
               strides=1,
               activation='relu')(input_layer)
lstm1 = LSTM(10, return_sequences=False)(conv1)
output_layer = Dense(1)(lstm1)
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=10, verbose=1,
                    validation_data=(X_valid, y_valid))


import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# prediction 
test_shape=X_test.shape[0]

X_test_pred = np.array(X_test).reshape(test_shape, time_steps, k_features)
y_test_pred = model.predict(X_test_pred)
y_test_pred = np.array(y_test_pred).reshape(test_shape)
np.square(np.subtract(y_test, y_test_pred)).mean() 

np.corrcoef(y_test, y_test_pred)

plt.plot(np.cumsum(y_test))
plt.plot(np.cumsum(y_test_pred))
y_test.mean()

# =============================================================================
# train 4  WaveNet-like 

import keras

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, k_features]))
for rate in (1, 2, 4, 8) * 2: # add pairs of layers using growing dilation rates: 1, 2, 4, 8, and again 1, 2, 4, 8
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="valid",
                                  activation="relu", dilation_rate=rate))
    # "causal" padding: this ensures that the convolutional layer does not peek into the future when
    # making predictions (it is equivalent to padding the inputs with the right amount of zeros on 
    # the left and using "valid" padding).
model.add(keras.layers. LSTM(16)) 
model.add(keras.layers.Dense(1))

print(model.summary()) 

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=10, verbose=1,
                    validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

# ----
# train 4a  WaveNet-like - VarianceScaling

init = keras.initializers.VarianceScaling(scale=.1, mode='fan_avg', distribution='uniform')

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[time_steps, k_features]))
for rate in (1, 2, 4, 8) * 2:
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="valid",
                                  activation="relu", dilation_rate=rate, kernel_initializer=init))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, kernel_initializer=init))

print(model.summary()) 

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=10, verbose=1,
                    validation_data=(X_valid, y_valid))
