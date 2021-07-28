import pandas as pd
import os
import jiagu

from data_paths import shapefile_path, hotspot_text_path, other_path
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
                              top_N_keywords_considered: int, save_filename: str,
                              save_path: str = other_path) -> list:
    """
    Use TextRank to extract keywords based on Weibos posted in a given day
    :param hotspot_dataframe: the weibo hotspot dataframe
    :param month: the studied month
    :param day: the studied day
    :param top_N_keywords_considered: top N keywords to consider
    :param save_filename: the name of the saved file having Weibos in a given day
    :param save_path: the path used to save the Weibos posted in a given day
    :return: a keyword list
    """
    if 'retweeters_text' not in hotspot_dataframe:
        dataframe_copy = hotspot_dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        dataframe_copy = hotspot_dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(
        lambda row: transform_datetime_string_to_datetime(row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['month'] = dataframe_copy.apply(lambda row: row['local_time'].month, axis=1)
    dataframe_copy['day'] = dataframe_copy.apply(lambda row: row['local_time'].day, axis=1)
    select_data = dataframe_copy.loc[(dataframe_copy['month'] == month) & (dataframe_copy['day'] == day)]
    text_list = list(select_data['text'])  # Here we only consider the original Weibo text
    repost_text_list = list(select_data['retweeters_text'])
    # Filter out the rows which don't have retweets
    repost_text_final = [text for text in repost_text_list if text != 'no retweeters']
    combined_text_list = text_list + repost_text_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    cleaned_text_string = [preprocessing_weibo(text, tokenization=False, return_word_list=False) for text in
                           combined_text_without_nan]
    text_string = " ".join(cleaned_text_string)
    keywords = jiagu.keywords(text_string, top_N_keywords_considered)
    final_keywords = [keyword for keyword in keywords if len(keyword) >= 2]
    select_data.to_csv(os.path.join(save_path, save_filename), encoding='utf-8')
    return final_keywords


def extract_keywords_in_a_month(hotspot_dataframe: pd.DataFrame, month: int,
                                top_N_keywords_considered: int, save_filename: str,
                                save_path: str = other_path) -> list:
    """
    Use TextRank to extract keywords based on Weibos posted in a given day
    :param hotspot_dataframe: the weibo hotspot dataframe
    :param month: the studied month
    :param top_N_keywords_considered: top N keywords to consider
    :param save_filename: the name of the saved file having Weibos in a given day
    :param save_path: the path used to save the Weibos posted in a hotspot in a given month
    :return: a keyword list
    """
    if 'retweeters_text' not in hotspot_dataframe:
        dataframe_copy = hotspot_dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        dataframe_copy = hotspot_dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(
        lambda row: transform_datetime_string_to_datetime(row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['month'] = dataframe_copy.apply(lambda row: row['local_time'].month, axis=1)
    select_data = dataframe_copy.loc[(dataframe_copy['month'] == month)]
    text_list = list(select_data['text'])  # Here we only consider the original Weibo text
    repost_text_list = list(select_data['retweeters_text'])
    # Filter out the rows which don't have retweets
    repost_text_final = [text for text in repost_text_list if text != 'no retweeters']
    combined_text_list = text_list + repost_text_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    print('We considered {} text strings.'.format(len(combined_text_without_nan)))
    cleaned_text_string = [preprocessing_weibo(text, tokenization=False, return_word_list=False) for text in
                           combined_text_without_nan]
    text_string = " ".join(cleaned_text_string)
    keywords = jiagu.keywords(text_string, top_N_keywords_considered)
    final_keywords = [keyword for keyword in keywords if len(keyword) >= 2]
    if 'csv' in save_filename:
        select_data.to_csv(os.path.join(save_path, save_filename), encoding='utf-8')
    elif 'xlsx' in save_filename:
        time_convert_dtype = {'local_time': str}
        select_data = select_data.astype(time_convert_dtype)
        select_data.to_excel(os.path.join(save_path, save_filename))
    else:
        raise ValueError('The file should be saved as either csv or xlsx.')
    return final_keywords
