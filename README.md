# Emoji Prediction

In this repository, we make our personal code for an emoji prediction task publicly available. However, please cite us, if you want to use our code for research, course work or any industry related product. Thank you!

We started this project as a course work for one of our first year's classes in the MSc IT and Cognition at the University of Copenhagen (UCPH). We aimed at predicting emojis in Twitter tweets solely given text data. In this study, owing to simplicity the emoji was required to be at the end of a Twitter tweet.

So far we achieved a solid 48% F1-score using a simple word n-gram text based approach with term frequency-inverse document frequency (tf-idf) weighting. Our machine learning algorithm of choice, was a Multi-layer Perceptron (MLP) with 50-60 hidden units. We implemented the MLP using Tensorflow's high-level deep learning library Keras.

In follow-up studies, we would like to develop a multi-modal classifier using both image and text data. In this endeavour, we will use dense word embeddings instead of sparse word vectors and deploy either Convolutional Neural Networks (CNNs) or LSTM-Recurrent Neural Networks (LSTMs). Stay tuned!
