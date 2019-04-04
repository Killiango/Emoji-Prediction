#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
import skimage.io as io

train_file = 'img_train_plaintext.txt'
test_file = 'img_test_plaintext.txt'
valid_file = 'img_valid_plaintext.txt'

def get_image(url):
    # downloads images as rgb images
    try:
        image = io.imread(url)
        return image
    except:
        return []
    
def filter_tweets(file):
    #This function removes tweets from the dataset where the image is no longer available
    #It also downloads all images and saves them into a seperate 'images' folder
    try:
        #Read data from file
        data = pd.read_csv(file, sep='\t', encoding = 'utf8', engine='c', header = 0)
        
        #array to save index for dropping them later
        rows_to_drop = []
        
        for i, tweet in data.iterrows():
            image = get_image(data.iloc[i, 1])
            
            if len(image) == 0 or len(data.iloc[i, 2].split(',')) > 1:
                rows_to_drop.append(i)
            else:
                # Save image as 'id.png'
                try:
                    io.imsave(fname= 'images\\'+ str(data.iloc[i, 0]) + '.png', arr=image)                
                except:
                    rows_to_drop.append(i)
                    print('Cannot save image: %s ' %(str(data.iloc[i, 0])))
        
        #Drop all lines for which there was no image available
        data = data.drop(data.index[rows_to_drop])
        
        #saves the filter dataset into a new file
        data.to_csv(path_or_buf= 'cleaned_' + file,index = False, sep = '\t', encoding='utf8')
        return 1
    
    except:
        print('Problem occured with the program')
        return 0         

print('\n---Start DL training set---')
filter_tweets(train_file)
print('---End DL training set---')
print('\n---Start DL test set---')
filter_tweets(test_file)
print('---End DL test set---')
print('\n---Start DL val set---')
filter_tweets(valid_file)
print('\n---End DL val set---')
