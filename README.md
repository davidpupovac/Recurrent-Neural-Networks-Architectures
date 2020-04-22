[![dep1](https://img.shields.io/badge/Python-3.7.3-brightgreen.svg)](https://www.python.org/) 
[![dep1](https://img.shields.io/badge/Tensorflow-1.14.0-brightgreen.svg)](https://www.tensorflow.org/)
[![dep1](https://img.shields.io/badge/Keras-2.2.4-brightgreen.svg)](https://keras.io/)

# Recurrent Neural Networks Architectures in Keras


Sequences data types (sentences, documents, audio samples, stock prices and etc. ) and the family of recurrent neural networks offer a variety of possible approaches.  There is a multitude of available architectures, RNN cell types and methods applicable to a variety of problems. 

The set of files in the repository demonstrates how to implement some of the available architectures  in addressing some of the most common types of problems. The existing solutions are predominantly demonstrated on regression type problems, but are easily amended for classification type problems (a limited number of classification problem examples is presented in files). 

The following problem families and solutions are  addressed in the files: 

#### [Multiple features, single time step -> single output feature, single time step]( https://github.com/davidpupovac/Recurrent-Neural-Networks-Architectures-in-Keras/blob/master/one-to_one_1.py)
- Simple Recurrent Neural Networks
- Long Short-Term Memory (LSTM) 
- LSTM with peephole connection
- Gated Recurrent Unit (GRU) 

#### [Multiple features, many time steps -> single output feature, single time step]( https://github.com/davidpupovac/Recurrent-Neural-Networks-Architectures-in-Keras/blob/master/many-to_one_1.py) 
- Long Short-Term Memory (LSTM) 
- Bidirectional LSTM
- Gated Recurrent Unit (GRU) 
- Convolutional/LSTM (Conv1D-LTSM)
- WaveNet (Convolutional layers with dilation rates)

#### [Multiple features, many time steps -> multiple output features, single time step](https://github.com/davidpupovac/Recurrent-Neural-Networks-Architectures-in-Keras/blob/master/many-to_one_2.py) 
- Long Short-Term Memory (LSTM) 
- Bidirectional LSTM
- Gated Recurrent Unit (GRU) 
- Convolutional/LSTM (Conv1D-LTSM)
- WaveNet (Convolutional layers with dilation rates) 

#### [Multiple features, many time steps -> single output feature, many time steps](https://github.com/davidpupovac/Recurrent-Neural-Networks-Architectures-in-Keras/blob/master/many-to_many_1.py)
- Long Short-Term Memory (LSTM) without TimeDistributed wrapper
- LSTM with TimeDistributed wrapper
- Gated Recurrent Unit (GRU) with TimeDistributed wrapper 
- Convolutional/ GRU (Conv1D- GRU)
- WaveNet (Convolutional layers with dilation rates) 
- Stateful  models



