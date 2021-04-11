import pandas as pd
import numpy as np
import os
import time
import jiagu
from collections import Counter
import paddlehub as hub

import data_paths
from process_text.text_preprocessing import preprocessing_weibo

# load the sentiment analysis module
senta = hub.Module(name='senta_bilstm')


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


# Load the pretrained sentiment analysis module
def get_sentiment_senta(text_string):
    """
    Get the sentiment analysis result
    :param text_string: a Weibo text string
    :return: 2 means positive, 1 means neutral, 0 means negative
    """
    cleaned_text = preprocessing_weibo(raw_tweet=text_string)
    sent_result = senta.sentiment_classify([cleaned_text])[0]
    positive_prob = sent_result['positive_probs']
    negative_prob = sent_result['negative_probs']
    if positive_prob > negative_prob and (positive_prob - negative_prob) > 0.3:
        return 2
    elif positive_prob < negative_prob and (negative_prob - positive_prob) > 0.3:
        return 0
    else:
        return 1


def baidu_sentiment(baidu_client, weibo_text:str) -> int:
    """
    Conduct the sentiment analysis using Baidu Sentiment Analysis API
    :param baidu_client: the Baidu sentiment analysis client
    :param weibo_text: a Weibo text
    :return: int, indicating the sentiment. 2 means positive; 1 means neutral; 0 means negative
    """
    print('Conducting the sentiment analysis of text: {}'.format(weibo_text))
    if weibo_text == 'no retweeters':
        return ()
    else:
        cleaned_text = preprocessing_weibo(raw_tweet=weibo_text, return_word_list=False)
        sent_result = baidu_client.sentimentClassify(cleaned_text)['items']
        positive_prob = sent_result[0]['positive_prob']
        negative_prob = sent_result[0]['negative_prob']
        sentiment = sent_result[0]['sentiment']
        seconds = np.random.randint(low=8, high=13)
        print('Sleep for {}'.format(seconds))
        time.sleep(seconds)
        return (sentiment, positive_prob, negative_prob)


def output_sentiment_int(sentiment_result: str):
    """
    Output the sentiment int result
    :param sentiment_result:
    :return:
    """
    if eval(sentiment_result)[0] == 'negative':
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
    if 'traffic_weibo' in dataframe:
        renamed_data = dataframe.copy()
    else:
        select_columns = ['index_val', 'author_id', 'weibo_id', 'created_at', 'text', 'lat', 'lon', 'retweeters',
                          'retweete_1', 'local_time', 'year', 'month', 'traffic_we', 'traffic_re', 'sent_weibo',
                          'sent_repos', 'Name', 'datatype', 'traffic_ty']
        rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text'}
        renamed_data = dataframe[select_columns].rename(columns=rename_dict)
    district_name_set_list = list(set(renamed_data[district_colname]))
    sent_district_dict = {key: [0, 0, 0] for key in district_name_set_list} # key: [positive, neutral, negative]
    for district_name, district_data in renamed_data.groupby(district_colname):
        sent_result = []
        for index, row in district_data.iterrows():
            weibo_sent = row['sent_weibo']
            sent_result.append(weibo_sent)
            if row['retweeters_text'] != 'no retweeters':
                repost_sent = row['sent_repos']
                sent_result.append(repost_sent)
        sent_counter = Counter(sent_result)
        sent_district_dict[district_name][0] = sent_counter[2] # positive
        sent_district_dict[district_name][1] = sent_counter[1] # neutral
        sent_district_dict[district_name][2] = sent_counter[0] # negative
    result_dataframe = pd.DataFrame(columns=['district_name', 'pos_count', 'neg_count', 'total_count', 'sent_index'])
    pos_count_list = []
    neg_count_list = []
    neutral_count_list = []
    for district_name in district_name_set_list:
        pos_count_list.append(sent_district_dict[district_name][0])
        neutral_count_list.append(sent_district_dict[district_name][1])
        neg_count_list.append(sent_district_dict[district_name][2])
    sum_count_array = np.array(pos_count_list) + np.array(neutral_count_list) + np.array(neg_count_list)
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
        # select dataframe in which the weibo is traffic relevant or repost is traffic relevant
        dataframe_selected = dataframe_copy[decision1 | decision2].reset_index(drop = True)
        print('{} rows have been selected.'.format(dataframe_selected.shape[0]))
        sentiment_weibo_list = []
        sentiment_reposts_list = []
        for _, row in dataframe_selected.iterrows():
            sentiment_weibo_list.append(get_sentiment_senta(row['text']))
            sentiment_reposts_list.append(get_sentiment_senta(row['retweeters_text']))
        dataframe_selected['sent_weibo'] = sentiment_weibo_list
        dataframe_selected['sent_repost'] = sentiment_reposts_list
        dataframe_selected.to_csv(os.path.join(data_paths.shanghai_jun_aug_traffic_sent, file[:-4]+'_sent.csv'),
                              encoding='utf-8')
        print('Done!')
        print('*'*10)


if __name__ == '__main__':

    # main_sent_analysis()
    combined_traffic_data = pd.read_csv(os.path.join(data_paths.weibo_data_path,
                                                  'combined_traffic_weibo_shanghai.csv'), encoding='utf-8', index_col=0)
    print(combined_traffic_data.head())
    sent_result_districts = compute_sentiment_across_districts(dataframe=combined_traffic_data, district_colname='Name')
    sent_result_districts.to_excel(os.path.join(data_paths.weibo_data_path, 'district_sent.xlsx'), encoding='utf-8')
