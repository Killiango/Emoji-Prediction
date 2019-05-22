from Classification import *
from Text_cleaning import *

import numpy as np
import time

# all hyper parameters
n_grams_hyper  = [(1,2), (2,2)] # (1,1)
min_df_hyper   = [2,3]

lowercase = True
use_stopwords = False
use_stemmer = False
use_sublinear_tf = True
use_SVD = False

dropout_hyper = [0.4, 0.5]
n_units_hyper  = [50, 60]
n_epoch = 8


# For saving the scores and hypers
all_hypers = []
all_MLP_scores = []
# loop over all hyperparameters
for n_gram in n_grams_hyper:
    for min_df in min_df_hyper:  
        for drop in dropout_hyper:
            # Text Cleaner
            X_train, X_val, X_test, y_train, y_val, y_test = text_cleaning(n_gram = n_gram,                    
                                                                           lower = lowercase,
                                                                           use_stopwords = use_stopwords, 
                                                                           stemmer = use_stemmer,
                                                                           min_df = min_df, 
                                                                           sublinear_tf = use_sublinear_tf, 
                                                                           use_SVD = use_SVD)
            print('Finished Text Cleaner')
            print('Train set dimensions: (%d, %d) ' %(X_train.shape))


            # Test a NN with various hidden units
            for n_unit in n_units_hyper:            
                hypers = (n_unit,n_gram,min_df,drop)
                all_hypers.append(hypers)

                print('Hyperparameters: ')
                print(hypers)
                print()

                MLP_scores = NNclassification(X_train, X_val, X_test, y_train, y_val, y_test,                                                                   n_units = n_unit, dropout = drop, n_epochs = n_epoch)

                all_MLP_scores.append(MLP_scores)
                
                
np.savetxt('Hyperparameters.txt', all_hypers)
np.savetxt('MLP-Scores.txt', all_MLP_scores)