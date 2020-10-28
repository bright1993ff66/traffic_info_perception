import pandas as pd
import os
from collections import Counter

import data_paths
from process_text.text_preprocessing import preprocessing_weibo


def count_words_freq(dataframe, text_column_name):
    """
    Count the word frequency based on a social media dataframe
    :param dataframe: a social medida dataframe, such as Weibo data
    :param text_column_name: the name of the text column which saves the social media messages
    :return: a dataframe with word frequency count
    """
    weibo_list = list(dataframe[text_column_name])
    cleaned_text_list = [preprocessing_weibo(weibo, return_word_list=True) for weibo in weibo_list]
    cleaned_text_list_final = []
    for cleaned_text in cleaned_text_list:
        cleaned_text_list_final.extend(cleaned_text)
    word_counter = Counter(cleaned_text_list_final)
    count_dataframe = pd.DataFrame(columns=['words', 'freq'])
    word_list = list(word_counter.keys())
    freq_list = [word_counter[word] for word in word_list]
    count_dataframe['words'] = word_list
    count_dataframe['freq'] = freq_list
    count_dataframe_sorted = count_dataframe.sort_values(by='freq', ascending=False).reset_index(drop=True)
    return count_dataframe_sorted


if __name__ =='__main__':
    data = pd.read_excel(os.path.join(data_paths.weibo_data_path, '1980308627_july_aug.xlsx'), index_col=0)
    count_dataframe = count_words_freq(data, text_column_name='text')
    print(count_dataframe.head(20))
    count_dataframe.to_excel(os.path.join(data_paths.weibo_data_path, 'word_freq_count.xlsx'), encoding='utf-8')




