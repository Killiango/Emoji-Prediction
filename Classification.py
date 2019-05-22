# coding: utf-8
import numpy as np
import keras
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

def get_model(hidden_units, input_dims, n_labels, dropout_r):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim = input_dims, activation = 'relu'))
    model.add(Dropout(dropout_r)) # dropout is important to prevent model from overfitting
    model.add(Dense(n_labels, activation = 'softmax'))
    adam = keras.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model
    

def probs_to_labels(y_probs):
    num_labels = [np.argmax(pred) for pred in y_probs]
    return num_labels


def accuracy_top_n(y_true, y_probs, top_n = 3):
    """
    If the correct label / emoji is among the top n (e.g., two, three) predictions,
    we consider the prediction as correctly labeled.
    """
    n_correct = 0
    n_total = 0
    
    for i, pred in enumerate(y_probs):
        top_3 = np.argsort(pred)[-top_n:]
        if y_true[i] in top_3:
            n_correct += 1
        n_total += 1
        
    ratio = n_correct / n_total
    return round(ratio, 4)

def NNclassification(X_train, X_val, X_test, y_train, y_val, y_test, n_units, dropout, n_epochs):
    n_batches=32

    # get indicator matrix with one-hot-encoded vectors per label (of all labels)
    y_train = to_cat_matrix(y_train)
    y_val = to_cat_matrix(y_val)

    MLP = get_model(n_units, X_train.shape[1], y_train.shape[1], dropout)
    print('Neural Net model constructed')
     

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val     = shuffle(X_val, y_val)
   
    
    MLP.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = n_epochs, 
          batch_size = n_batches, callbacks = [es, mc])
    
    
    # load best model
    saved_model = load_model('best_model.h5')

    # get predictions
    y_probs = saved_model.predict(X_test)
    
    # convert predictions to labels
    y_preds = probs_to_labels(y_probs)

    # calculate scores
    f1 = f1_score(y_true=y_test, y_pred=y_preds, average='weighted')
    acc_top1 = accuracy_top_n(y_test, y_probs, 1)
    acc_top3 = accuracy_top_n(y_test, y_probs, 3)
    
    print(classification_report(y_test, y_preds, target_names = list([str(i) for i in range(0, 10)])))
    
    return (f1, acc_top1, acc_top3)
