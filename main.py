from Classification import *
from Text_cleaning import *

import numpy as np
import time

# all hyper parameters
n_grams_hyper  = [(1,1), (1,2), (2,2)]
lowercase = True
max_df = 1
min_df_hyper   = [2,3,4]
use_stopwords = False
use_stemmer = False
use_sublinear_tf = True
use_SVD = True

n_units_hyper  = [20, 30, 50, 75]
n_epochs_hyper = [5,10,15]

n_epoch = 10

all_hypers = []
all_MLP_scores = []
all_Log_Reg_scores = []
all_MNB_scores = []

# loop over all hyperparameters
for n_gram in n_grams_hyper:
    for min_df in min_df_hyper:  
        # Text Cleaner
        X_train, X_val, X_test, y_train, y_val, y_test = text_cleaning(n_gram = n_gram,                    
                                                                       lower = lowercase,
                                                                       use_stopwords = use_stopwords, 
                                                                       stemmer = use_stemmer,
                                                                       max_df = max_df,
                                                                       min_df = min_df, 
                                                                       sublinear_tf = use_sublinear_tf, 
                                                                       use_SVD = use_SVD)
        print('Finished Text Cleaner')
        
        print('Train set dimensions: (%d, %d) ' %(X_train.shape))
        # Machine Learning Classification
        Log_Reg_scores, SVM_scores = classification(X_train, X_val, X_test, y_train, y_val, y_test)
            
        print('Logistic Regression scores: \n ', Log_Reg_scores)
        print('SVM scores: \n ', SVM)
        
        # Test a NN with various hidden units
        for n_unit in n_units_hyper:            
            hypers = (n_unit, n_gram,min_df)
            all_hypers.append(hypers)
            
            print('Hyperparameters: ')
            print(hypers)
            print()
            
            MLP_scores = NNclassification(X_train, X_val, X_test, y_train, y_val, y_test,                                                                   n_units = n_unit, n_epochs = n_epoch)
                     
            all_MLP_scores.append(MLP_scores)
            all_Log_Reg_scores.append(Log_Reg_scores)
            all_MNB_scores.append(MNB_scores)        
            
np.savetxt('Hyperparameters.txt', all_hypers)
np.savetxt('MLP-Scores.txt', all_MLP_scores)
np.savetxt('LogReg-Scores.txt', all_Log_Reg_scores)
np.savetxt('MNB-Scores.txt', all_MNB_scores)