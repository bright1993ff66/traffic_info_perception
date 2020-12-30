import pandas as pd
import os
import jiagu

from data_paths import shapefile_path, hotspot_text_path
from process_text.text_preprocessing import preprocessing_weibo
from utils import transform_datetime_string_to_datetime, timezone_shanghai


def extract_keywords(dataframe, traffic_type: str, top_N_keywords_considered: int) -> list:
    """
    Extract the keywords using textrank for a traffic Weibo dataframe
    :param dataframe: a traffic Weibo dataframe
    :param traffic_type: the type of traffic information to be considered: ['accident', 'congestion', 'condition']
    :param top_N_keywords_considered: the number of keywords output by textrank
    :return: a list of keywords(length >= 2) extracted from a traffic Weibo dataframe
    """
    assert traffic_type in ['accident', 'congestion', 'condition'], 'The traffic information type is not right!'
    data_select = dataframe.loc[dataframe['traffic_ty'] == traffic_type]
    if 'retweeters_text' not in data_select:
        data_select_renamed = data_select.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        data_select_renamed = data_select.copy()
    text_list = list(data_select_renamed['text'])  # Here we only consider the original Weibo text
    repost_text_list = list(data_select_renamed['retweeters_text'])
    repost_text_final = [text for text in repost_text_list if text != 'no retweeters']
    combined_text_list = text_list + repost_text_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    cleaned_text_string = [preprocessing_weibo(text, tokenization=False, return_word_list=False) for text in
                           combined_text_without_nan]
    text_string = " ".join(cleaned_text_string)
    keywords = jiagu.keywords(text_string, top_N_keywords_considered)
    final_keywords = [keyword for keyword in keywords if len(keyword) >= 2]
    return final_keywords


def extract_keywords_in_a_day(hotspot_dataframe: pd.DataFrame, month: int, day: int,
                              top_N_keywords_considered: int) -> list:
    """
    Use TextRank to extract keywords based on Weibos posted in a given day
    :param hotspot_dataframe: the weibo hotspot dataframe
    :param month: the studied month
    :param day: the studied day
    :param top_N_keywords_considered: top N keywords to consider
    :return: a keyword list
    """
    dataframe_copy = hotspot_dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(
        lambda row: transform_datetime_string_to_datetime(row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['month'] = dataframe_copy.apply(lambda row: row['local_time'].month, axis=1)
    dataframe_copy['day'] = dataframe_copy.apply(lambda row: row['local_time'].day, axis=1)
    select_data = dataframe_copy.loc[(dataframe_copy['month'] == month) & (dataframe_copy['day'] == day)]
    text_list = list(select_data['text'])  # Here we only consider the original Weibo text
    repost_text_list = list(select_data['retweeters_text'])
    repost_text_final = [text for text in repost_text_list if text != 'no retweeters']
    combined_text_list = text_list + repost_text_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    cleaned_text_string = [preprocessing_weibo(text, tokenization=False, return_word_list=False) for text in
                           combined_text_without_nan]
    text_string = " ".join(cleaned_text_string)
    keywords = jiagu.keywords(text_string, top_N_keywords_considered)
    final_keywords = [keyword for keyword in keywords if len(keyword) >= 2]
    return final_keywords


def keyword_extract_hotspot_main(top_N):
    """Main function to extract keywords"""
    hotspot1_data = pd.read_csv(os.path.join(shapefile_path, 'traffic_hotspot1.txt'), encoding='utf-8', index_col=0)
    hotspot2_data = pd.read_csv(os.path.join(shapefile_path, 'traffic_hotspot2.txt'), encoding='utf-8', index_col=0)

    hotspot1_keywords_accident = extract_keywords(hotspot1_data, traffic_type='accident',
                                                  top_N_keywords_considered=top_N)
    hotspot1_keywords_congestion = extract_keywords(hotspot1_data, traffic_type='congestion',
                                                    top_N_keywords_considered=top_N)
    print('The keywords of accidents in hotspot 1 are: {}'.format(hotspot1_keywords_accident))
    print('The keywords of congestions in hotspot 1 are: {}'.format(hotspot1_keywords_congestion))

    hotspot2_keywords_accident = extract_keywords(hotspot2_data, traffic_type='accident',
                                                  top_N_keywords_considered=top_N)
    hotspot2_keywords_congestion = extract_keywords(hotspot2_data, traffic_type='congestion',
                                                    top_N_keywords_considered=top_N)
    print('The keywords of accidents in hotspot 2 are: {}'.format(hotspot2_keywords_accident))
    print('The keywords of congestions in hotspot 2 are: {}'.format(hotspot2_keywords_congestion))


def keyword_extract_in_given_days():
    """
    Extract the keywords in some given days
    :return:
    """
    acc_dataframe = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_acc_hotspot_with_sent.csv'),
                                encoding='utf-8', index_col=0)
    cgs_dataframe = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_cgs_hotspot_with_sent.csv'),
                                   encoding='utf-8', index_col=0)
    hotspot_days = {'acc': [(6, 22), (7, 23), (8, 28)], 'cgs': [(7, 11), (7, 20), (8, 8)]}
    for traffic_type in hotspot_days:
        if traffic_type == 'acc':
            print('='*10)
            print('Generating keywords for the accident relevant Weibos')
            studied_time_list = hotspot_days[traffic_type]
            for day_tuple in studied_time_list:
                print('For the date: {}-{}:'.format(day_tuple[0], day_tuple[1]))
                keywords = extract_keywords_in_a_day(hotspot_dataframe=acc_dataframe, month=day_tuple[0],
                                                     day=day_tuple[1], top_N_keywords_considered=30)
                print(keywords)
            print('=' * 10)
        else:
            print('=' * 10)
            print('Generating keywords for the congestion relevant Weibos')
            studied_time_list = hotspot_days[traffic_type]
            for day_tuple in studied_time_list:
                print('For the date: {}-{}:'.format(day_tuple[0], day_tuple[1]))
                keywords = extract_keywords_in_a_day(hotspot_dataframe=cgs_dataframe, month=day_tuple[0],
                                                     day=day_tuple[1], top_N_keywords_considered=30)
                print(keywords)
            print('=' * 10)


if __name__ == '__main__':
    keyword_extract_in_given_days()

