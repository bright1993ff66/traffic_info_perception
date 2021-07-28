from nltk.stem.lancaster import LancasterStemmer
import re
import os
import string
import zh_core_web_sm
import emoji
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

import data_paths

# Get all the punctuations
punctuations_all = string.punctuation + '。，﹑？！：；“”（）《》•……【】'

# Load the Chinese language model for preprocessing
nlp = zh_core_web_sm.load()


def remove_emojis(text) -> str:
    """
    Remove emojis from text
    :param text: a text list or text string
    :return: cleaned text list or cleaned text string based on the type of text
    """
    assert type(text) == list or type(text) == str, 'The input should be either list or string'
    if type(text) == list:
        return [c for c in text if c not in emoji.UNICODE_EMOJI]
    else:
        return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)


def create_augment_text_file(dataframe, text_colname, label_colname, save_path, save_filename):
    """
    Create the txt file for weibo data augmentation
    :param dataframe: a pandas dataframe saving the Weibo text
    :param text_colname: the colname of the column saving the Weibo text
    :param label_colname: the colname of the column saving the Weibo annotated label
    :param save_path: the saving path
    :param save_filename: the saved filename
    :return: None. The txt file waiting to be augmented is saved to local directory
    """
    text_list = list(dataframe[text_colname])
    label_list = list(dataframe[label_colname])
    with open(os.path.join(save_path, save_filename), 'w', encoding='utf-8') as file:
        for text, label in zip(text_list, label_list):
            file.write(str(label) + '\t' + str(text) + '\n')


def create_stopwords_list(stopword_path: str) -> list:
    """
    Create the Chinese stopword list
    :param stopword_path: the path which contains the stopword
    :return: a Chinese stopword list
    """
    stopwords_list = []
    with open(os.path.join(stopword_path, 'hit_stopwords.txt'), 'r', encoding='utf-8') as stopword_file:
        for line in stopword_file:
            line = line.replace("\r", "").replace("\n", "")
            stopwords_list.append(line)
    return stopwords_list


def preprocessing(raw_tweet, stemming=False, remove_stopwords=False):
    """
    Preprocess the tweet: consider hashtag words as just words and pass them to linguistic modules
    :param raw_tweet: the tweet text to be processed
    :param stemming: conduct word stemming or not
    :param remove_stopwords: whether remove stopwords or not
    :return: cleaned tweet
    """
    # 0. Remove the urls
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)" \
            r"))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text_without_url = re.sub(regex, '', raw_tweet)

    # 1. Only consider letters and numbers
    letters_nums_only = re.sub("[^a-zA-Z0-9]", " ", text_without_url)
    #
    # 2. Convert to lower case, split into individual words
    words = letters_nums_only.split()

    # 4. Remove stop words
    if remove_stopwords:
        stops = set(STOP_WORDS)
        meaningful_words = [w for w in words if not w in stops]
    else:
        meaningful_words = words
    #
    # 5. Stemming
    if stemming:
        st = LancasterStemmer()
        text_stemmed = [st.stem(word) for word in meaningful_words]
        result = text_stemmed
    else:
        result = meaningful_words

    return " ".join(result)


def preprocessing_weibo(raw_tweet, tokenization=True, return_word_list=True):
    """
    Preprocess the Chinese weibo
    :param raw_tweet: raw Chinese weibo string
    :param tokenization: whether we conduct the tokenization or not
    :param return_word_list: whether we return the word list or not
    :return:
    """
    # 0. Remove the urls
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)" \
            r"))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text_without_url = re.sub(regex, '', raw_tweet)

    # 1. Remove the @user patterns
    text_without_username1 = re.sub('@[\u4e00-\u9fa5a-zA-Z0-9_-]{4,30}', '', text_without_url)
    text_without_username2 = re.sub('\/\/@.*?:', '', text_without_username1)

    # 2. Don't consider the numbers, punctuations
    stopwords_list = create_stopwords_list(stopword_path=data_paths.chinese_stopword_path)
    words_only = re.sub("[0-9]", "", text_without_username2)
    words_without_puc = re.sub("[{}]".format(punctuations_all), "", words_only)

    # 3. Tokenize the weibos using spacy
    if tokenization:
        doc = nlp(words_without_puc)
        stopwords_list.append(' ')
        words_list = [token.lemma_ for token in doc]
        words_final = [word for word in words_list if word not in stopwords_list]

        if return_word_list:
            return [re.sub(r'[^\u4e00-\u9fff]+', '', chinese_words) for chinese_words in words_final]
        else:
            return re.sub(r'[^\u4e00-\u9fff ]+', '', ' '.join(words_final))
    else:
        return words_only


def preprocessing_traffic_accounts(traffic_weibo):
    # 0. Remove the urls
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)" \
            r"))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text_without_url = re.sub(regex, '', traffic_weibo)

    # 1. Don't consider the numbers
    words_only = re.sub("[0-9]", "", text_without_url)

    return words_only


if __name__ == '__main__':

    sample_text = 'no retweeters'
    result = preprocessing_weibo(raw_tweet=sample_text, return_word_list=True)
    print(result)
