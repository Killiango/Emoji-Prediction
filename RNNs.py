import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, TimeDistributed

class GRU_NET():
    
    embedding_dim = 300
    
     def __init__(self, vocab_size: int, hidden_units: int, n_features: int, n_timesteps, n_labels: int, dropout = False, optimizer):
        self.vocab_size = vocab_size
        self.hidden_units = hidden_units
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.n_labels = n_labels
        self.loss = "sparse_categorical_crossentropy" # if we want to predict word vecs instead of labels, use cosine proximity
        self.dropout = dropout
        self.optimizer = optimizer
        
        # 2 GRU hidden layers with n hidden units
        # 1 output layer with softmax as activation function (output = probability distribution over the labels)
        print('Build model...')
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim, input_shape = (n_samples, n_timesteps)))
        self.model.add(GRU(hidden_units, activation='relu', recurrent_activation='hard_sigmoid', return_sequences = True))       
        self.model.add(GRU(hidden_units, activation='relu', recurrent_activation='hard_sigmoid', return_sequences = True))
        
        if self.dropout != False:
            self.model.add(Dropout(dropout))
        
        self.model.add(TimeDistributed(Dense(n_labels, activation = 'softmax')))
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = ['accuracy'])
                       
    def fit(self, X_train, y_train, n_epochs, n_batches):
        return self.model.fit(Xtrain, ytrain, epochs = n_epochs, batch_size = n_batches)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class LSTM_NET():
    
    embedding_dim = 300
    
    def __init__(self, vocab_size: int, hidden_units: int, n_features: int, n_timesteps = None, n_labels: int, dropout = False, optimizer):
        self.vocab_size = vocab_size
        self.hidden_units = hidden_units
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.n_labels = n_labels
        self.loss = "sparse_categorical_crossentropy" # if we want to predict emoji vecs instead of emoji labels, use cosine proximity
        self.dropout = dropout
        self.optimizer = optimizer
        
        # 2 LSTM hidden layers with n hidden units
        # 1 output layer with softmax as activation function (output = probability distribution over the labels)
        print('Build model...')
        self.model = Sequential()
        ### EMBEDDING -> MASKING ZERO SET TO TRUE for variable timesteps
        self.model.add(Embedding(vocab_size, embedding_dim, input_shape = (n_samples, n_timesteps)))
        self.model.add(LSTM(hidden_units, activation = 'relu', recurrent_activation = 'hard_sigmoid', return_sequences = True))
        
        if self.dropout != False:
            self.model.add(Dropout(self.dropout))
        
        self.model.add(LSTM(hidden_units, activation = 'relu', recurrent_activation = 'hard_sigmoid', return_sequences = True))
        
        if self.dropout != False:
            self.model.add(Dropout(self.dropout))
        
        self.model.add(TimeDistributed(Dense(n_labels, activation = 'softmax')))
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = ['accuracy'])
        
    def fit(self, X_train, y_train, n_epochs, n_batches):
        return self.model.fit(Xtrain, ytrain, epochs = n_epochs, batch_size = n_batches)
    
    def predict(self, X_test):
        return self.model.predict(X_test)