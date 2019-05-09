from Classification import *
from Text_cleaning import *

import numpy as np
import time

# all hyper parameters


n_units_hyper  = [20, 30, 50, 75]
n_grams_hyper  = [(1,1), (1,2), (2,2)]
min_df_hyper   = [2,3,4]
n_epochs_hyper = [5,10,15]

n_epoch = 10

all_hypers = []
all_MLP_scores = []
all_Log_Reg_scores = []
all_MNB_scores = []

# loop over all hyperparameters
for n_unit in n_units_hyper:
    for n_gram in n_grams_hyper:
        for min_df in min_df_hyper:
            hypers = (n_unit, n_gram,min_df)
            all_hypers.append(hypers)
            
            # In the innermost loop
            print('Hyperparameters: ')
            print(hypers)
            print()
            
            start_time = time.time()
            # Text Cleaner
            X_train, X_val, X_test, y_train, y_val, y_test = text_cleaning(
                n_gram = n_gram, min_df = min_df, lower = True, use_stopwords = False, stemmer = False)

            
            print('Finished Text Cleaner')
            
            # Machine Learning Classification
            MLP_scores, Log_Reg_scores, MNB_scores = classification(X_train, X_val, X_test, y_train, y_val, y_test,
                                                                   n_units = n_unit,
                                                                   n_epochs= n_epoch,
                                                                   fit_prior=True)
            
            end_time = time.time()
            print('Time elapsed for this run: ', end_time-start_time)
                  
            all_MLP_scores.append(MLP_scores)
            all_Log_Reg_scores.append(Log_Reg_scores)
            all_MNB_scores.append(MNB_scores)
            
            break
        break
    break
            
np.savetxt('Hyperparameters.txt', all_hypers)
np.savetxt('MLP-Scores.txt', all_MLP_scores)
np.savetxt('LogReg-Scores.txt', all_Log_Reg_scores)
np.savetxt('MNB-Scores.txt', all_MNB_scores)