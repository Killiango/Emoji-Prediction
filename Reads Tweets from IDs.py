import getopt
import logging
import os
import sys
import tweepy
import pandas as pd 
import numpy as np
import re


# Generate your own at https://apps.twitter.com/app
CONSUMER_KEY = 'q8svcQ1GKW2yknY8MCZLvcO7w'
CONSUMER_SECRET = 'kk9eMhfIMVxoDEoKR63ddWooW87Ya7IgUt5oC31S0TpAXeiMdh'
OAUTH_TOKEN = '917762487608659970-G1v4Nr01JQA9UKqO1HP4g4bPwKT7LAr'
OAUTH_TOKEN_SECRET = 'p1Zp4ophwRbRvR5yET3ppXWWg7fEshIyWwby9vTBxR9CF'

# connect to twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth)

# batch size depends on Twitter limit, 100 at this time
batch_size = 100

#Some emojis have character length of more than 1
emoji_threshold = 3  

#The files containing the datasets
train_file = 'cleaned_img_train_plaintext.txt'
test_file = 'cleaned_img_test_plaintext.txt'
valid_file = 'cleaned_img_valid_plaintext.txt'



def locate_emoji(emoji_pattern, text: str):
    emoji = ''.join(emoji_pattern.findall(text))
    try:
        index = text.index(emoji)
    except:
        index = -emoji_threshold
    return emoji, index

def get_tweets(twapi, file):
    data = pd.read_csv(file, sep='\t', encoding = 'utf8', engine='c', header = 0)   
    '''
    Fetches content for tweet IDs in a file using bulk request method,
    which vastly reduces number of HTTPS requests compared to above;
    however, it does not warn about IDs that yield no tweet.
    `twapi`: Initialized, authorized API object from Tweepy
    '''
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        "]+",
        flags=re.UNICODE)

    tweet_ids = data.id.values.tolist()
    emoji_labels = data.annotations.values.tolist()

    all_tweets = []
    labels = []
    i = 0  #for debug
    # process list of ids until it's empty
    while len(tweet_ids) > 0:
        if len(tweet_ids) < batch_size:
            tweets = twapi.statuses_lookup(
                id_=tweet_ids, include_entities=False, trim_user=True)
            tweet_ids = []
        else:
            tweets = twapi.statuses_lookup(
                id_=tweet_ids[:batch_size],
                include_entities=False,
                trim_user=True)
            tweet_ids = tweet_ids[batch_size:]

        for tweet in tweets:

            #removes the link of the tweet
            text = re.sub(r'http\S+', '', tweet.text).strip(' ')
            
            #Remove tweets where emoji is not at the end
            emoji, index = locate_emoji(emoji_pattern, text)
            
            if index >= len(text) - emoji_threshold:
                #removes the emojis from the text
                text = emoji_pattern.sub(r'', text).strip(' ')

                #then appends the tweet and emoji to our final dataset
                all_tweets.append(np.array([text]))
                labels.append(emoji)
        
        #For debuging
        i += 1
        if i == 20:
            break
            
    features = np.array(all_tweets)
    #TODO: add the text preprocessing function here
   
    labels = np.array(labels)
    return features, labels

print('\nBegin dl train tweets')
Xtrain, ytrain = getTweets(api, train_file)
np.savetxt('tweets_train.txt', Xtrain)
np.savetxt('emojis_train.txt', ytrain)

print('\nBegin dl test tweets')
Xtest, ytest = getTweets(api, test_file)
np.savetxt('tweets_test.txt', Xtest)
np.savetxt('emojis_test.txt', ytest)

print('\nBegin dl val tweets')
Xval, yval = getTweets(api, val_file)
np.savetxt('tweets_val.txt', Xval)
np.savetxt('emojis_val.txt', yval)









