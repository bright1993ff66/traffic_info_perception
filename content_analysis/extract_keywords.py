import pandas as pd
import os
import jiagu

from data_paths import shapefile_path, hotspot_text_path
from process_text.text_preprocessing import preprocessing_weibo


def extract_keywords(dataframe, traffic_type: str, top_N_keywords_considered: int):
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


if __name__ == '__main__':
    # keyword_extract_hotspot_main(top_N=20)
    acc_dataframe = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_acc_hotspot_with_sent.csv'),
                                encoding='utf-8', index_col=0)
    acc_keywords = extract_keywords(dataframe=acc_dataframe, traffic_type='accident',
                                    top_N_keywords_considered=30)
    print('The keywords for the accident data are: {}'.format(acc_keywords))
    conges_dataframe = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_cgs_hotspot_with_sent.csv'),
                                   encoding='utf-8', index_col=0)
    conges_keywords = extract_keywords(dataframe=conges_dataframe, traffic_type='congestion',
                                       top_N_keywords_considered=30)
    print('The keywords for the congestion data are: {}'.format(conges_keywords))

