# coding: utf-8
import numpy as np
import keras
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense, Dropout


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

def get_model(input_dims, hidden_units, n_labels):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim = input_dims, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n_labels, activation = 'softmax'))
    adam = keras.optimizers.Adam(lr=0.001, beta_1 = 0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model
    

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


def NNclassification(X_train, X_val, X_test, y_train, y_val, y_test, n_units, n_epochs):
    n_batches=32

    # get indicator matrix with one-hot-encoded vectors per label (of all labels)
    y_train = to_cat_matrix(y_train)
    y_val = to_cat_matrix(y_val)
    y_test = to_cat_matrix(y_test)

    MLP = get_model(X_train.shape[1], n_units, y_train.shape[1])
    print('Neural Net model constructed')
    
    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val     = shuffle(X_val, y_val)
 
    MLP.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = n_epochs, batch_size = n_batches)
    
    
    # Get predictions on the test set
    y_probs_MLP = MLP.predict(X_test)
    y_preds_MLP = probs_to_labels(y_probs_MLP)
    f1_MLP = f1_score(y_true=y_test, y_pred=y_preds_MLP, average='micro')
    acc_MLP = accuracy_score(y_test, y_preds_MLP)
    
    return (f1_MLP, acc_MLP)
    
def classification(X_train, X_val, X_test, y_train, y_val, y_test):
    
    
    # Build 3 machine learning models: Multi-layer Perceptron (MLP), Logistic Regression, Support Vector Machine (SVM)
    LogReg = LogisticRegression(penalty='l2', class_weight='balanced', random_state=42, solver='lbfgs', max_iter=50, multi_class='multinomial')
    SVM = SVC(gamma = 'auto', kernel = 'linear')
    print('Sk learn classifiers Constructed')

    
    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val     = shuffle(X_val, y_val)
 
    # Train the models (and validate using the Val set)
    LogReg.fit(X_train, y_train)
    SVM.fit(X_train, y_train)
    
    # Get predictions for the test set
    y_probs_LogReg = LogReg.predict(X_test)
    y_preds_LogReg = probs_to_labels(y_probs_LogReg)
    f1_LogReg = f1_score(y_true=y_test, y_pred=y_preds_LogReg, average='micro')
    acc_LogReg = accuracy_score(y_test, y_preds_LogReg)

    y_probs_SVM = SVM.predict(X_test)
    y_preds_SVM = probs_to_labels(y_probs_SVM)
    f1_SVM = f1_score(y_true=y_test, y_pred=y_preds_SVM, average='micro')
    acc_SVM = accuracy_score(y_test, y_preds_SVM)

    
    return (f1_LogReg, acc_LogReg), (f1_SVM, acc_SVM)
