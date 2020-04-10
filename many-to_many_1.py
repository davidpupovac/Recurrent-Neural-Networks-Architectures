# Python version: 3.7.3 
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
"""

                 Many (time steps) to many (time steps) 1  
                
 (Multiple features, many time steps -> single output feature, many time steps)
 
                                
"""
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# =============================================================================
# =============================================================================
# =============================================================================
# Example  of regression problem


# =============================================================================
# Simulate some toy data 

import numpy as np

np.random.seed(15)

num_samples = 1000 # number of samples
time_steps = 20  # number of time steps 
k_features=6 # number of features

def gen_autocor_seq (time_steps, cor_coef, scale=1):
    series = np.random.normal(scale=scale, size=time_steps)
    for j in range(2,time_steps):
        series[j] = cor_coef * series[j-1] + np.random.normal(scale=scale)
        return series

def data_gen (num_samples,time_steps, k_features):
    X = np.empty([num_samples, time_steps, k_features])
    y = np.empty([num_samples, time_steps])
    for i in range(0,num_samples):
        for j in range(0,k_features):
            X[i, :, j]= gen_autocor_seq(time_steps,0.65) # arbitrary autocorrelation coefficient=0.65
            for k in range(0,time_steps):
                y[i, k]= sum(X[i, k, :])
    return X, y

X,y=data_gen(num_samples, time_steps, k_features)

train_size = int(np.around([len(X)*0.7]))
valid_size = int(np.around([len(X)*0.2]))

X_train, y_train = X[:train_size,:,:], y[:train_size] 
X_valid, y_valid = X[train_size: train_size+valid_size,:,:], y[train_size:train_size+valid_size]
X_test, y_test = X[train_size+valid_size:len(X)], y[train_size+valid_size:len(X)]

# =============================================================================
# train 1 - Solution via Simple LSTM - does not uses the TimeDistributed wrapper 

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, k_features)))
model.add(Dense(time_steps)) 

model.compile(optimizer='adam', loss='mse')
print(model.summary()) 

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# predict from the model
y_pred = model.predict(X_test[0:1,:,:]) # prediction for a single sample 
y_actu = np.array(X_test[0:1,:,:]).reshape(time_steps, k_features).sum(1)

np.corrcoef(y_actu, y_pred) # see the correlation between the actual and predicted values.

# ------------
# train 2 - Solution via Stacked LSTM - does not use the TimeDistributed wrapper 

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, k_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(time_steps))

model.compile(optimizer='adam', loss='mse')
print(model.summary())    

history = model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# =============================================================================
# train 3 - LSTM with TimeDistributed wrapper

# ---
# data - you have to reshape the output feature
y_train = np.array(y_train).reshape(train_size, time_steps, 1) 
y_valid = np.array(y_valid).reshape(valid_size, time_steps, 1)

# ---
from tensorflow import keras 

model = keras.models.Sequential([
    keras.layers.LSTM(40, activation='relu', return_sequences=True, input_shape=[time_steps, k_features]),
    # you can specify input shape as: input_shape=[None, k_features]
    keras.layers.LSTM(40, activation='relu', return_sequences=True), 
    # TimeDistributed requires return_sequences=True
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])
print(model.summary())    

model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01))

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# ---
y_pred = model.predict(X_test[0:1,:,:]) # prediction for a single case
y_actu = np.array(X_test[0:1,:,:]).reshape(time_steps, k_features).sum(1)
np.corrcoef(y_actu, np.array(y_pred).reshape(time_steps, ))


# =============================================================================
# train 4 - GRUs - with TimeDistributed wrapper

model = keras.models.Sequential([
    keras.layers.GRU(20, return_sequences=True, input_shape=[time_steps, k_features]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])
print(model.summary())    

model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01))

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# =============================================================================
# train 5 - Convolutional Layers 

model = keras.models.Sequential([
    # as input has time_steps and so does the output, you have to keep dimensions the same:
    # If you use a 1D convolutional layer with a stride of 1 and "same" padding,
    # then the output sequence will have the same length as the input sequence 
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=1, padding="same",
                        input_shape=[None, k_features]), 
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])

print(model.summary())    

model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01))

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# =============================================================================
# train 6 - Wave Net-like 

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, k_features]))
for rate in (1, 2, 4, 8, 10): 
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",strides=1, 
                                  activation="relu", dilation_rate=rate))
model.add(keras.layers. LSTM(16, return_sequences=True)) 
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))

print(model.summary())    

model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01))

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# =============================================================================
# train 7 - Bidirectional

from keras.layers import Bidirectional
from keras.layers import TimeDistributed

model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, k_features))))
model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True)))
model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mse', lr=0.01)
# print(model.summary()) # this will not work here you need to fit or build the model

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=True)

# =============================================================================
# =============================================================================
# =============================================================================

# Stateful  models

# For a stateful RNN it is necessary to use SEQUENTIAL and NONOVERLAPPING input sequences 
# (rather than the shuffled and overlapping sequences we used to train stateless RNNs). 
# Thus, we must not call the shuffle() method. 

# -----
# Simulate some toy data

np.random.seed(15)

time_steps = 20000  # number of time steps 
k_features=6 # number of features

def data_gen (time_steps, k_features):
    X = np.empty([time_steps, k_features])
    y = np.empty([time_steps])
    for j in range(0,k_features):
        X[:, j]= gen_autocor_seq(time_steps,0.65) # arbitrary autocorrelation coefficient=0.65
        for k in range(0,time_steps):
                y[k]= sum(X[k, :])
    return X, y

X,y=data_gen(time_steps, k_features)

# redefine time_steps ()
time_steps = 20
num_samples = 1000

X = np.array(X).reshape(num_samples, time_steps, k_features)
y = np.array(y).reshape(num_samples, time_steps, 1)

train_size = int(np.around([len(X)*0.7]))
valid_size = int(np.around([len(X)*0.2]))

X_train, y_train = X[:train_size,:,:], y[:train_size] 
X_valid, y_valid = X[train_size: train_size+valid_size,:,:], y[train_size:train_size+valid_size]
X_test, y_test = X[train_size+valid_size:len(X)], y[train_size+valid_size:len(X)]

y_train = np.array(y_train).reshape(train_size, time_steps, 1)
y_valid = np.array(y_valid).reshape(valid_size, time_steps, 1)

# =============================================================================
# train 8 - Stateful GRU (stateful across batches and epochs)

batch_size = 20 # you must have inputs with a number of samples that can be divided by the batch size

model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, stateful=True,                     
                     batch_input_shape=[batch_size, time_steps, k_features]),
    # batch_input_shape argument in the first layer must be specified; we can leave 
    # the second dimension unspecified since the inputs could have any length
    keras.layers.GRU(128, return_sequences=True, stateful=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1)) 
])


model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01))

history = model.fit(X_train, y_train, epochs=50, batch_size=20, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=False)

# =============================================================================
# train 9 - Stateful GRU (stateful across batches)

# In addition dropout:

# All recurrent layers (except for keras.layers.RNN) and all cells provided by Keras have 
# a dropout hyperparameter and a recurrent_dropout hyperparameter: the former defines the
# dropout rate to apply to the inputs (at each time step), and the latter defines the dropout
# rate for the hidden states (also at each time step). 

batch_size = 20 

model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, stateful=True,
                     dropout=0.2, recurrent_dropout=0.2, # dropout
                      batch_input_shape=[batch_size, time_steps, k_features]),
    keras.layers.GRU(128, return_sequences=True, stateful=True,
                     dropout=0.2, recurrent_dropout=0.2), # dropout
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])

# ---
# train
    
# The stateful model will save states across batches and epochs.
# To reset state at every epoch set the following: 
class ResetStatesCallback(keras.callbacks.Callback): 
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01))
history = model.fit(X_train, y_train, epochs=50, batch_size=20, verbose=1,
                    validation_data=(X_valid, y_valid), callbacks=[ResetStatesCallback()])

# or just do:
for epoch in range(50):
	model.fit(X_train, y_train, epochs=1, batch_size=20, verbose=1,
                    validation_data=(X_valid, y_valid), shuffle=False)
	model.reset_states()

# ---
# prediction

# After this model is trained, it will only be possible to use it to make predictions for batches of
# the same size as were used during training. To avoid this restriction, create an identical
# stateless model, and copy the stateful model’s weights to this model.

stateless_model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[time_steps, k_features]),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])

import tensorflow as tf    
    
stateless_model.build(tf.TensorShape([None, None, k_features]))
stateless_model.set_weights(model.get_weights())

model = stateless_model

# predict from the model:
y_pred = model.predict(X_test[0:1,:,:])
y_actu = np.array(X_test[0:1,:,:]).reshape(time_steps, k_features).sum(1)
np.corrcoef(y_actu, np.array(y_pred).reshape(time_steps, ))
