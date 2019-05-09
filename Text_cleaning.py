
# coding: utf-8
import numpy as np
import pandas as pd
import nltk
import os
import re
import sys

from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

from sklearn.feature_extraction.text import CountVectorizer



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

def emoji_to_int(labels: list, emoji_map):
    return [emoji_map[emoji] for emoji in labels]

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

# Potential problem: if min_df > 1, then maybe we should also filter all tweets that have less than 2 words of the vocab???
def tweets_cleaning(tweets, labels, train = False, use_stopwords = False, use_unigrams = True, use_bigrams = False, lowercase = True, stemming = False, min_df = 1):
    """
    Text cleaning function that performs all necessary text preprocessing steps.
    Function only keeps characters, that are alphanumerical (non-alphanumerical values are discarded).
    Digits are treated by regular expressions.
    Lower-casing is performed to reduce noise and normalize the text (convert it into a uniform representation).
    Stemming is performed to only keep the stem of each word token but not any other deviated form. 
    Stop words (i.e., words that occur more frequently than other words in a given corpus) are removed.
    """

     # initialize Lancaster stemmer
    if stemming:
        st = LancasterStemmer()

    if use_stopwords:
        stop_words = list(set(stopwords.words('english'))) 
        
    cleaned_data = []
    cleaned_labels = []

    bigrams_dict = dict()
    vocab = dict()

    for tweet, label in zip(tweets, labels):
        tweet = re.sub(r'&amp\S+','', tweet)
        tweet = re.sub(r' & ', ' and ', tweet)
        tweet = re.sub(r'!+', ' ! ', tweet)
        tweet = re.sub(r'[?]+', ' ? ', tweet)
        tweet = re.sub('@.+', '@user', tweet)
        tweet = re.sub('#', '# ', tweet)

        # Create spaces instead of some punctuation marks, but not if it's part of an emoticon
        tweet = ' '.join([word if re.search(r'(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S)+', word)
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

            if len(cleaned_word) > 0:
                if use_stopwords and cleaned_word not in stopwords:
                    cleaned_tweet.append(cleaned_word)
                else:
                    cleaned_tweet.append(cleaned_word)
                    
                if train:
                    if cleaned_word in vocab:
                        vocab[cleaned_word] += 1
                    else:
                        vocab[cleaned_word] = 1

        # only append tweets with more than 1 word per tweet
        if len(cleaned_tweet) > 1:

            if train and use_bigrams:

                bigrams = [' '.join([cleaned_tweet[i-1], cleaned_tweet[i]]) for i, _ in enumerate(cleaned_tweet) if i > 0]

                for bigram in bigrams:

                    if bigram in bigrams_dict:
                        bigrams_dict[bigram] += 1
                    else:
                        bigrams_dict[bigram] = 1 

            cleaned_tweet = ' '.join(cleaned_tweet)
            cleaned_data.append(cleaned_tweet)
            cleaned_labels.append(label)

    if train:
        vocab = [word for word, freq in vocab.items() if freq >= min_df]

        if use_bigrams:
            all_bigrams = [bigram for bigram, freq in bigrams_dict.items() if freq >= min_df]
            vocab.extend(all_bigrams)
            
            if not use_unigrams:
                return cleaned_data, cleaned_labels, sorted(all_bigrams)

    assert len(cleaned_data) == len(cleaned_labels)

    return cleaned_data, cleaned_labels, sorted(vocab)    

def bag_of_words(train: list, test: list, val: list, ngram: tuple, vocab = None):
    """
    Create a count (!) based bag-of-words unigram or bigram representation of provided tweets.
    Ngram is set to unigram by default. If bigram bag-of-words should be created, pass tuple (2, 2).

    Vocabulary argument is set to None by default. 
    You can pass a vocabulary to this function, which may then be used for CountVectorizer. 
    If you do not pass a vocabulary to this function, CountVectorizer will create a vocabulary itself.
    """ 

    # initialize vectorizer (word-ngram representation)
    vectorizer = CountVectorizer(encoding = 'utf-8', ngram_range = ngram, analyzer = 'word', vocabulary = vocab)
    train_BoW = vectorizer.fit_transform(train).toarray()
    test_BoW = vectorizer.transform(test).toarray()
    val_BoW = vectorizer.transform(val).toarray()

    return train_BoW, test_BoW, val_BoW

def text_cleaning(n_gram: tuple, min_df: int, lower = True, use_stopwords = False, stemmer = False):
    train_proc = pd.read_csv('train_set_processed.csv')
    val_proc = pd.read_csv('val_set_processed.csv')
    test_proc = pd.read_csv('test_set_processed.csv')

    use_unigrams = True if n_gram[0] == 1 else False
    use_bigrams  = True if n_gram[1] == 2 else False    
    
    top_10_test = count_emojis(test_proc)
    emoji_map = {emoji: index for index,emoji in enumerate(top_10_test)}

    
    train_data = keep_top_10(train_proc, top_10_test)
    test_data = keep_top_10(test_proc, top_10_test)
    val_data = keep_top_10(val_proc, top_10_test)

    
    cleaned_train_data, train_labels, vocab = tweets_cleaning(train_data.text,                                                      
                                                        train_data.label, 
                                                        train = True,
                                                        use_stopwords = use_stopwords,
                                                        use_unigrams = use_unigrams, 
                                                        use_bigrams = use_bigrams,                                                       
                                                        lowercase = lower,                                                               
                                                        min_df = min_df)

    cleaned_test_data, test_labels, _ = tweets_cleaning(test_data.text, 
                                                           test_data.label, 
                                                           use_stopwords = use_stopwords,
                                                           lowercase = lower)

    cleaned_val_data, val_labels, _  = tweets_cleaning(val_data.text, 
                                                         test_data.label, 
                                                         use_stopwords = use_stopwords,
                                                         lowercase = lower)


    print("Number of Tweets per data set after text cleaning was computed:\n")
    print("Train: {}\n".format(len(cleaned_train_data)))
    print("Test: {}\n".format(len(cleaned_test_data)))
    print("Validation: {}\n".format(len(cleaned_val_data)))
    print("Number of unique tokens in the vocabulary: {}\n".format(len(vocab)))

    y_train = emoji_to_int(train_labels, emoji_map)
    y_test = emoji_to_int(test_labels,emoji_map)
    y_val = emoji_to_int(val_labels,emoji_map)


    X_train, X_test, X_val = bag_of_words(cleaned_train_data, cleaned_test_data, cleaned_val_data, ngram = n_gram, vocab = vocab)

    return X_train, X_val, X_test, y_train, y_val, y_test