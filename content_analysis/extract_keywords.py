import pandas as pd
import os
import jiagu

from visualizations import generate_wordcloud
from data_paths import shapefile_path
from process_text.text_preprocessing import preprocessing_weibo


def extract_keywords(dataframe, traffic_type:str, top_N_keywords:int):
    """
    Extract the keywords using textrank for a traffic Weibo dataframe
    :param dataframe: a traffic Weibo dataframe
    :param traffic_type: the type of traffic information to be considered: ['accident', 'congestion', 'condition']
    :param top_N_keywords: the number of considered keywords
    :return: a list of keywords extracted from a traffic Weibo dataframe
    """
    assert traffic_type in ['accident', 'congestion', 'condition'], 'The traffic information type is not right!'
    data_select = dataframe.loc[dataframe['traffic_ty'] == traffic_type]
    text_list = list(data_select['text']) # Here we only consider the original Weibo text
    cleaned_text_string = [preprocessing_weibo(text, tokenization=False, return_word_list=False) for text in text_list]
    text_string = " ".join(cleaned_text_string)
    keywords = jiagu.keywords(text_string, top_N_keywords)
    return keywords


def keyword_extract_main(top_N):
    """Main function to extract keywords"""
    hotspot1_data = pd.read_csv(os.path.join(shapefile_path, 'traffic_hotspot1.txt'), encoding='utf-8', index_col=0)
    hotspot2_data = pd.read_csv(os.path.join(shapefile_path, 'traffic_hotspot2.txt'), encoding='utf-8', index_col=0)

    hotspot1_keywords_accident = extract_keywords(hotspot1_data, traffic_type='accident', top_N_keywords=top_N)
    hotspot1_keywords_congestion = extract_keywords(hotspot1_data, traffic_type='congestion', top_N_keywords=top_N)
    print('The keywords of accidents in hotspot 1 are: {}'.format(hotspot1_keywords_accident))
    print('The keywords of congestions in hotspot 1 are: {}'.format(hotspot1_keywords_congestion))

    hotspot2_keywords_accident = extract_keywords(hotspot2_data, traffic_type='accident', top_N_keywords=top_N)
    hotspot2_keywords_congestion = extract_keywords(hotspot2_data, traffic_type='congestion', top_N_keywords=top_N)
    print('The keywords of accidents in hotspot 2 are: {}'.format(hotspot2_keywords_accident))
    print('The keywords of congestions in hotspot 2 are: {}'.format(hotspot2_keywords_congestion))


if __name__ == '__main__':
    keyword_extract_main()
