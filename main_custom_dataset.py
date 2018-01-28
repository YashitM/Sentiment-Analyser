"""
 *  Author : YashitM 
 *  Created On : Sat Jan 27 2018
 *  File : main_custom_dataset.py
"""
import numpy
import pandas as pd
import tweepy
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier


def get_filtered_words(sentence):
    stop_words = set(stopwords.words('english'))
    filtered_words = [item for item in sentence if item not in stop_words]
    filtered_words = dict([(word, True) for word in filtered_words])
    return filtered_words


def get_positive_sentences_custom():
    f = open("custom_dataset/pos.txt", "r").readlines()
    pos = []
    for i in f:
        words = word_tokenize(i)
        pos.append((get_filtered_words(words), "positive"))
    return pos


def get_negative_sentences_custom():
    f = open("custom_dataset/neg.txt", "r").readlines()
    neg = []
    for i in f:
        words = word_tokenize(i)
        neg.append((get_filtered_words(words), "negative"))
    return neg


def main(sentence):
    tokenized_sentence = word_tokenize(sentence)
    filtered_sentence = get_filtered_words(tokenized_sentence)

    negative_reviews = get_negative_sentences_custom()
    positive_reviews = get_positive_sentences_custom()

    data = negative_reviews + positive_reviews

    classifier = NaiveBayesClassifier.train(data)
    print(classifier.classify(filtered_sentence))


if __name__ == '__main__':
    sentence = "Hello guys how are you, I am good!"
    main(sentence)
