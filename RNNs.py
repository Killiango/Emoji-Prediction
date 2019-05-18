import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, TimeDistributed, Flatten

### IMPORTANT NOTE: Initialize the weights of the embedding layer with pre-trained word embeddings (e.g., word2vec, GloVe)
### Trainable needs to be set to false, if we pass pre-trained word embeddings such as word2vec or GloVe

class GRU_NET():

    #embedding_dim = 300
    
    def __init__(self, vocab_size: int, max_length: int, hidden_units: int, n_features: int, embedding_matrix, n_labels: int, 
        optimizer, dropout = False):
        self.vocab_size = vocab_size
        self.max_length = max_length # max document length across the entire corpus
        self.hidden_units = hidden_units
        self.n_features = n_features
        self.embedding_matrix = embedding_matrix
        self.n_labels = n_labels
        self.loss = "categorical_crossentropy" # if we want to predict emoji vecs instead of emoji labels, use cosine proximity
        self.optimizer = optimizer
        self.dropout = dropout
        
        # 2 GRU hidden layers with n hidden units
        # 1 output layer with softmax as activation function (output = probability distribution over the labels)
        print('Build model...')
        self.model = Sequential()
        
        self.model.add(Embedding(vocab_size + 2, n_features, weights = [embedding_matrix], 
                                 trainable = False, input_length = max_length, mask_zero = True))
        
        self.model.add(GRU(hidden_units, activation='relu', recurrent_activation='hard_sigmoid', 
                           return_sequences = True))    
        if self.dropout:
            self.model.add(Dropout(dropout))
        
        self.model.add(GRU(hidden_units, activation='relu', recurrent_activation='hard_sigmoid', 
                           return_sequences = False))
        
        if self.dropout:
            self.model.add(Dropout(dropout))
        
        #self.model.add(TimeDistributed(Dense(self.n_labels, activation = 'softmax')))
        self.model.add(Dense(self.n_labels, activation = 'softmax'))
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = ['accuracy'])
                       
    def fit(self, X_train, y_train, X_val, y_val,  n_epochs, n_batches):
        return self.model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = n_epochs, batch_size = n_batches)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class LSTM_NET():

    def __init__(self, vocab_size: int, max_length: int, hidden_units: int, n_features: int, embedding_matrix, n_labels: int, 
        optimizer, dropout = False):
        self.vocab_size = vocab_size
        self.max_length = max_length # max document length across the entire corpus
        self.hidden_units = hidden_units
        self.n_features = n_features
        self.embedding_matrix = embedding_matrix
        self.n_labels = n_labels
        self.loss = "categorical_crossentropy" # if we want to predict emoji vecs instead of emoji labels, use cosine proximity
        self.optimizer = optimizer
        self.dropout = dropout
        
        # 2 LSTM hidden layers with n hidden units
        # 1 output layer with softmax as activation function (output = probability distribution over the labels)
        print('Build model...')
        self.model = Sequential()

        self.model.add(Embedding(vocab_size + 2, n_features, weights = [embedding_matrix], input_length = max_length,
                                 trainable = False, mask_zero = True))
        

        self.model.add(LSTM(hidden_units, activation = 'relu', recurrent_activation = 'hard_sigmoid',
                            return_sequences = True))
        
        if self.dropout:
            self.model.add(Dropout(self.dropout))
        
        self.model.add(LSTM(hidden_units, activation = 'relu', recurrent_activation = 'hard_sigmoid', 
                        return_sequences = False))
        
        if self.dropout:
            self.model.add(Dropout(self.dropout))
        
        #self.model.add(Flatten())
        self.model.add(TimeDistributed(Dense(self.n_labels, activation = 'softmax')))
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = ['accuracy'])
        
    def fit(self, X_train, y_train, X_val, y_val, n_epochs, n_batches):
        return self.model.fit(Xtrain, ytrain, validation_data = (X_val, y_val), epochs = n_epochs, batch_size = n_batches)
    
    def predict(self, X_test):
        return self.model.predict(X_test)