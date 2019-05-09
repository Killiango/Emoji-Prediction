#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import getopt
import logging
import nltk
import os
import re
import sys
import tweepy

from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


# In[2]:


train_proc = pd.read_csv('train_set_processed.csv')
val_proc = pd.read_csv('val_set_processed.csv')
test_proc = pd.read_csv('test_set_processed.csv')


# In[3]:


def count_emojis(data, n = 10):
    """
    Function that counts the number of emojis in the data set.
    Display the n most frequent emojis.
    """
    emoji_counts = {}
    for index, row in data.iterrows():
        emoji = row[1]
        if emoji not in emoji_counts:
            # compute simultaneous counting
            emoji_counts[emoji] = data[data.label == emoji].count()[1]
            
    # sort emojis by freq in descending order (list of tuples will be returned)
    sorted_emoji_counts = sorted(emoji_counts.items(), key= lambda kv: kv[1], reverse=True)
        
    return [emoji[0] for emoji in sorted_emoji_counts[:n]]


# In[4]:


top_10_test = count_emojis(test_proc)
print(top_10_test)


# In[5]:


emoji_map = {emoji: index for index,emoji in enumerate(top_10_test)}
def emoji_to_int(labels:list):
    return [emoji_map[emoji] for emoji in labels]


# In[6]:


def keep_top_10(data, top_10: list): 
    """
    Function that checks, whether Tweet consists of one of the top ten emojis.
    If, and only if, Tweet consists one of the most frequent emojis, 
    Tweet will be used for further analysis.
    Else: Line will be dropped.
    """
    idx_drop = []
    for index, row in data.iterrows():
        if row[1] not in top_10:
            idx_drop.append(index)
    return data.drop(data.index[idx_drop])


# In[7]:


train_data = keep_top_10(train_proc, top_10_test)
print(len(train_data))


# In[8]:


test_data = keep_top_10(test_proc, top_10_test)
print(len(test_data))


# In[9]:


val_data = keep_top_10(val_proc, top_10_test)
print(len(val_data))


# In[10]:


# create list of stopwords
stop_words = list(set(stopwords.words('english')))


# In[11]:


def tweets_cleaning(tweets, labels, stopwords: list, train = False, lowercase = True, stemming = False, min_df = 1):
    """
    Text cleaning function that performs all necessary text preprocessing steps.
    Function only keeps characters, that are alphanumerical (non-alphanumerical values are discarded).
    Digits are treated by regular expressions.
    Lower-casing is performed to reduce noise and normalize the text (convert it into a uniform representation).
    Stemming is performed to only keep the stem of each word token but not any other deviated form. 
    Stop words (i.e., words that occur more frequently than other words in a given corpus) are removed.
    """
    
     # initialize Lancaster stemmer, if stemming is set to True
    if stemming:
        st = LancasterStemmer()
    
    cleaned_data = []
    cleaned_labels = []
    vocab = {}
    
    for tweet, label in zip(tweets,labels):
        tweet = re.sub(r'&amp\S+','', tweet)
        tweet = re.sub(r' & ', ' and ', tweet)
        tweet = re.sub(r'!!*', ' ! ', tweet)
        tweet = re.sub(r'[?]+', ' ? ', tweet)
        tweet = re.sub('@.+', '@user', tweet)
        tweet = re.sub('#', '# ', tweet)

        # Create spaces instead of some punctuation marks, but not if it's part of an emoticon
        tweet = ' '.join(
            [word if re.search(r'(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S)+', word)
            else re.sub('[,.;\-_:/\n\t]+', ' ', word) for word in tweet.split()])
        
        tweet = tweet.split(" ")
        
        cleaned_tweet = []
        for word in tweet:
            
            #if emoticon is in word, keep the emoticon
            if re.search(r'(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S)+', word):
                cleaned_word = word
            else:
                # keep special characters which might carry important information
                # perform lower-casing to normalize the text and reduce noise
                cleaned_word = ''.join([char for char in word if re.search('[<>$#€£!?@=]', char) or
                                        char.isalnum()])
            if lowercase:
                cleaned_word = cleaned_word.lower()
                
            if "<3" not in cleaned_word:
                cleaned_word = re.sub('[0-9]', '0', cleaned_word)
  
            # removes each \n (i.e., new line) or \t (i.e., tab) -> pipe char denotes a disjunction
            cleaned_word = re.sub(r'( \n| \t)+', '', cleaned_word)
            
            if stemming:
                cleaned_word = st.stem(cleaned_word)
                        
            if len(cleaned_word) > 0: # and cleaned_word not in stopwords:
                cleaned_tweet.append(cleaned_word)
                if train:
                    if cleaned_word in vocab:
                        vocab[cleaned_word] += 1
                    else:
                        vocab[cleaned_word] = 1
            
        # only append tweets with more than 1 word per tweet
        if len(cleaned_tweet) > 1:
            cleaned_tweet = ' '.join(cleaned_tweet)
            cleaned_data.append(cleaned_tweet)
            cleaned_labels.append(label)
    
    if train:
        vocab = [word for word, freq in vocab.items() if freq >= min_df]
    
    assert len(cleaned_data) == len(cleaned_labels)
    return cleaned_data, cleaned_labels, sorted(vocab)

# Potential problem: if min_df > 1, then maybe we should also filter all tweets that have less than 2 words of the vocab???


# In[12]:


lower = True 
cleaned_train_data, train_labels, vocab = tweets_cleaning(train_data.text, train_data.label, stop_words,train=True, lowercase=lower, min_df=2)
cleaned_test_data, test_labels, _ = tweets_cleaning(test_data.text, test_data.label, stop_words, lowercase=lower)
cleaned_val_data, val_labels, _= tweets_cleaning(val_data.text, test_data.label, stop_words, lowercase=lower)


# In[13]:


print(len(cleaned_train_data))
print(len(cleaned_test_data))
print(len(cleaned_val_data))


# In[14]:


print(len(vocab))
#print(vocab)


# In[15]:


y_train = emoji_to_int(train_labels)
y_test = emoji_to_int(test_labels)
y_val = emoji_to_int(val_labels)


# ### Functions for the Bag of Words approach

# In[16]:


def bag_of_words(train: list, test: list, val: list, ngram:tuple, vocab = None):
    """
    Create a count (!) based bag-of-words unigram or bigram representation of provided tweets.
    Ngram is set to unigram by default. If bigram bag-of-words should be created, pass tuple (2, 2).
    
    Vocabulary argument is set to None by default. 
    You can pass a vocabulary to this function, which may then be used for CountVectorizer. 
    If you do not pass a vocabulary to this function, CountVectorizer will create a vocabulary itself.
    """ 
    
    # initialize vectorizer (word-ngram representation)
    vectorizer = CountVectorizer(encoding = 'utf-8',ngram_range = ngram, analyzer = 'word', vocabulary=vocab)
    train_BoW = vectorizer.fit_transform(train).toarray()
    test_BoW = vectorizer.transform(test).toarray()
    val_BoW = vectorizer.transform(val).toarray()
    
    return train_BoW, test_BoW, val_BoW


# In[17]:


X_train, X_test, X_val = bag_of_words(cleaned_train_data, cleaned_test_data, cleaned_val_data, ngram = (1,2), vocab=vocab)


# In[19]:


np.savetxt('X_train_BoW.txt', X_train)
np.savetxt('X_test_BoW.txt', X_test)
np.savetxt('X_val_BoW.txt', X_val)

np.savetxt('y_train_BoW.txt', y_train)
np.savetxt('y_test_BoW.txt', y_test)
np.savetxt('y_val_BoW.txt', y_val)


# ### Functions for the Embeddings approach

# In[ ]:


def get_embeddings(text_file):

    """ 
    Read GloVe txt.-file, load pre-trained word embeddings into memory
    and create a word_to_embedding dictionary, where keys are the discrete word strings
    and values are the corresponding continuous word embeddings, retrieved from the GloVe txt.-file.
    For unkown words, the representation is an empty vector (i.e., zeros matrix).
    """
    embeddings_dict = {}

    with open(text_file, encoding="utf8") as file:

        for line in file:
            values = line.split()
            word = values[0]
            wordvec = np.array(values[1:], dtype = 'float32')
            embeddings_dict[word] = list(wordvec)

    return embeddings_dict


# In[ ]:


emoji_embeddings = get_embeddings("emoji2vec.txt")


# In[ ]:


def get_emojivecs(emoji_embeddings: dict, corpus: list, dims: int):

    N = len(corpus)
    M = dims
    
    emojivecs = []
    
    # document = tweet; corpus = all tweets
    for emoji in corpus:
        emoji_sequence = []

        try:
            emojivec = emoji_embeddings[emoji]
            assert len(emojivec) == M
            emoji_sequence.append(emojivec)
        except KeyError:
            emoji_sequence.append([0 for _ in range(M)])
            print("This {} does not exist in the pre-trained emoji embeddings.".format(emoji))

        emojivecs.append(emoji_sequence)

    assert len(emojivecs) == N
    return np.array(emojivecs)


# In[ ]:


def get_wordvecs(word_embeddings: dict, corpus: list, dims: int, zeros_padding = False):

    """ 
    Return a concatenated word vector representation of each tweet.
    The concatenated word vectors serve as the input data for the LSTM RNN.
    Each word (embedding) denotes a time step. (Number of timesteps is equal to the length of the input sentence.)
    
    Check whether length of word vector is equal to the number of dimensions we pass to this function.
    For unknown words (i.e., if key does not exist), the representation is an empty vector / zeros matrix of len dims.

    Sequences can have variable length (i.e., number of time steps per batch).
    However, in some cases you might want to zero pad the batch if a sequence < max length of sequences in the corpus.
    By default this argument is set to False as Keras and Tensorflow except input sequences of variable length.
    If set to True, zero padding is computed.
    """

    N = len(corpus)
    M = dims
    global max_length
    max_length = max([len(sequence) for sequence in corpus])
    wordvecs_corpus = []
    
    # document = tweet; corpus = all tweets
    for document in corpus:
        wordvec_sequence = []
        for word in document:
            
            try:
                wordvec = word_embeddings[word]
                assert len(wordvec) == M
                wordvec_sequence.append(wordvec)
            except KeyError:
                wordvec_sequence.append([0 for _ in range(M)])
                
        # needs to be resolved (!)
        if zeros_padding == True: 
            if len(document) < max_length:

                for _ in range(len(document), max_length):
                    wordvec_sequence.append([0 for _ in range(M)])

                assert len(wordvec_sequence) == max_length
        wordvecs_corpus.append(wordvec_sequence)

    assert len(wordvecs_corpus) == N
    return np.array(wordvecs_corpus)


# In[ ]:


from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('word2vec.txt', binary=False)


# In[ ]:


word_embeddings = get_embeddings("word2vec.txt")

