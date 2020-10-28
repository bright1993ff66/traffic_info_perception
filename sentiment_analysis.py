import pandas as pd
import numpy as np
import os
import jiagu
from collections import Counter

import data_paths
from process_text.text_preprocessing import preprocessing_weibo


def get_sentiment(cleaned_string: str, return_prob=True):
    """
    Get the sentiment of a Weibo. 1: positive; 0: negative
    :param cleaned_string: the cleaned Weibo string
    :param return_prob: whether return the probability of being positive or negative
    """
    sent_tuple = jiagu.sentiment(cleaned_string)
    if return_prob: return sent_tuple
    if sent_tuple[0] == 'negative':
        return 0
    else:
        return 1


def compute_sentiment_across_districts(dataframe, district_colname):
    """
    Compute the sentiment index across districts
    :param dataframe: the Weibo dataframe
    :param district_colname: the district name
    :return: the sentiment index across districts
    """
    district_name_set_list = list(set(dataframe[district_colname]))
    sent_district_dict = {key: [0, 0] for key in district_name_set_list}
    for district_name, district_data in dataframe.groupby(district_colname):
        sent_result = []
        sent_counter = Counter()
        for index, row in district_data.iterrows():
            weibo_sent = eval(row['sent_weibo'])[0]
            sent_result.append(weibo_sent)
            if row['retweeters_text'] != 'no retweeters':
                repost_sent = eval(row['sent_repos'])[0]
                sent_result.append(repost_sent)
            sent_counter = Counter(sent_result)
        sent_district_dict[district_name][0] = sent_counter['positive']
        sent_district_dict[district_name][1] = sent_counter['negative']
    result_dataframe = pd.DataFrame(columns=['district_name', 'pos_count', 'neg_count', 'total_count', 'sent_index'])
    pos_count_list = []
    neg_count_list = []
    for district_name in district_name_set_list:
        pos_count_list.append(sent_district_dict[district_name][0])
        neg_count_list.append(sent_district_dict[district_name][1])
    sum_count_array = np.array(pos_count_list) + np.array(neg_count_list)
    sent_index_array = (np.array(neg_count_list) - np.array(pos_count_list))/sum_count_array
    result_dataframe['district_name'] = district_name_set_list
    result_dataframe['pos_count'] = pos_count_list
    result_dataframe['neg_count'] = neg_count_list
    result_dataframe['total_count'] = sum_count_array.tolist()
    result_dataframe['sent_index'] = sent_index_array.tolist()
    result_dataframe_sorted = result_dataframe.sort_values(by='total_count', ascending=False).reset_index(drop=True)
    return result_dataframe_sorted


def main_sent_analysis():
    """
    Conduct the sentiment analysis for traffic-related Weibos
    :return:
    """
    for file in os.listdir(data_paths.shanghai_jun_aug_traffic):
        print('*'*15)
        print('Conducting the sentiment analysis of the file: {}'.format(file))
        dataframe = pd.read_csv(os.path.join(data_paths.shanghai_jun_aug_traffic, file), encoding='utf-8', index_col=0)
        dataframe_copy = dataframe.copy()
        decision1 = (dataframe_copy['traffic_weibo'].isin([1,2]))
        decision2 = (dataframe_copy['traffic_repost'].isin([1,2]))
        dataframe_selected = dataframe_copy[decision1 | decision2].reset_index(drop = True)
        print('{} rows have been selected.'.format(dataframe_selected.shape[0]))
        sentiment_weibo_list = []
        sentiment_reposts_list = []
        for _, row in dataframe_selected.iterrows():
            cleaned_weibo = preprocessing_weibo(raw_tweet=str(row['text']), return_word_list=False)
            cleaned_reposts = preprocessing_weibo(raw_tweet=str(row['retweeters_text']), return_word_list=False)
            sentiment_weibo_list.append(get_sentiment(cleaned_string=cleaned_weibo, return_prob=True))
            sentiment_reposts_list.append(get_sentiment(cleaned_string=cleaned_reposts, return_prob=True))
        dataframe_selected['sent_weibo'] = sentiment_weibo_list
        dataframe_selected['sent_repost'] = sentiment_reposts_list
        dataframe_selected.to_csv(os.path.join(data_paths.shanghai_jun_aug_traffic_sent, file[:-4]+'_sent.csv'),
                              encoding='utf-8')
        print('Done!')
        print('*'*10)



if __name__ == '__main__':

    print('Conduct the sentiment analysis for all the districts...')
    combined_traffic_data = pd.read_csv(os.path.join(data_paths.weibo_data_path, 'combined_traffic_weibo_shanghai.csv'),
                                        encoding='utf-8', index_col=0)
    sent_result_districts = compute_sentiment_across_districts(dataframe=combined_traffic_data, district_colname='Name')
    sent_result_districts.to_csv(os.path.join(data_paths.weibo_data_path, 'district_sent.csv'), encoding='utf-8')
