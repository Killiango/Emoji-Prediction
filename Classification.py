
# coding: utf-8

# In[4]:


import numpy as np
import keras
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[3]:


X_train = np.loadtxt('X_train_BoW.txt')
X_test = np.loadtxt('X_test_BoW.txt')
X_val = np.loadtxt('X_val_BoW.txt')

y_train = np.loadtxt('y_train_BoW.txt')
y_test = np.loadtxt('y_test_BoW.txt')
y_val = np.loadtxt('y_val_BoW.txt')

print('Finished loading data! \n')


# In[10]:


def to_cat_matrix(y):
    """ 
    Binary one-hot encoding using an indicator matrix.
    This function converts labels to a categorical matrix which is of size N x K.
    Each row is a row vector with k-1 zeros and a single 1.
    """
    N = len(y)
    K = len(set(y))
    ind_matrix = np.zeros((N,K), dtype = int)
    for i, cat in enumerate(y):
        ind_matrix[i, int(cat)] = 1
    return ind_matrix

# get indicator matrix with one-hot-encoded vectors per label (of all labels)
y_train = to_cat_matrix(y_train)
y_val = to_cat_matrix(y_val)
y_test = to_cat_matrix(y_test)


# In[12]:


def get_model(hidden_units: int, input_dims: int, n_labels: int):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim = input_dims, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n_labels, activation = 'softmax'))
    adam = keras.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model


# In[21]:


def probs_to_labels(y_probs):
    num_labels = [np.argmax(pred) for pred in y_probs]
    return num_labels

def accuracy_score(ytrue, ypred):
    n_correct = 0
    n_total = 0
    for i, pred in enumerate(ypred):
        try:
            if pred == ytrue[i]:
                n_correct += 1
        except:
            if np.argmax(pred) == np.argmax(ytrue[i]):
                n_correct += 1
        n_total += 1
    ratio = n_correct / n_total
    accuracy = ratio * 100
    return round(accuracy, 2)


# ## Start actual model

# In[14]:


# set number of hidden units, epochs and batch size
n_units = 50
n_epochs = 10
n_batches = 32


# In[24]:


model = get_model(n_units, X_train.shape[1], 10)
print('Model Constructed')


# In[16]:


model.fit(X_train, y_train, validation_data=(X_val, y_val) epochs = n_epochs, batch_size = n_batches)


# In[25]:


# get predictions
y_probs_test = model.predict(X_test)
y_preds_test = probs_to_labels(y_probs_test)

f1= f1_score1(y_true=y_test, y_pred=y_preds_test, average='micro')
print('F1 Score: %.3f' % (f1))

