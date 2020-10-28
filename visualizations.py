# For bacis
import os
import pandas as pd
import numpy as np
from collections import Counter

# For plotting
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

from data_paths import other_path, figures_path, weibo_data_path, shapefile_path, random_seed
from process_text.text_preprocessing import preprocessing_weibo
from utils import transform_datatime_string_to_datetime
from time_analysis import timezone_shanghai


def generate_wordcloud(dataframe, text_column, save_filename):
    """
    Generate the wordcloud based on a Weibo dataframe
    :param dataframe: a Weibo dataframe
    :param text_column: the considered text column
    :param save_filename: the filename of the saved wordcloud plot
    """
    font_path = os.path.join(other_path, 'SourceHanSerifK-Light.otf')
    wc = WordCloud(font_path=font_path, background_color='white', max_words=50000, max_font_size=300,
                   random_state=random_seed)
    text_list = list(dataframe[text_column])
    cleaned_text = ' '.join([preprocessing_weibo(text) for text in text_list])
    wc.generate_from_text(cleaned_text)
    figure, axis = plt.subplots(1, 1, dpi=200)
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    figure.savefig(os.path.join(figures_path, save_filename))


def generate_wordcloud_hotspots(hotspot_data, text_column, accident_filename, congestion_filename):
    """
    Generate the wordcloud for the Weibo text found in one hotspot
    :param hotspot_data: Weibo data found in one traffic hotspot
    :param text_column: the text column to be considered
    :param accident_filename: the filename of the accident wordcloud
    :param congestion_filename: the filename of the congestion wordcloud
    :return:
    """
    print('For this traffic hotspot, we have got {} weibos'.format(hotspot_data.shape[0]))
    accident_data = hotspot_data.loc[hotspot_data['traffic_ty'] == 'accident']
    congestion_data = hotspot_data.loc[hotspot_data['traffic_ty'] == 'congestion']
    generate_wordcloud(dataframe=accident_data, text_column=text_column, save_filename=accident_filename)
    generate_wordcloud(dataframe=congestion_data, text_column=text_column, save_filename=congestion_filename)


def create_hour_weekday_plot(dataframe:pd.DataFrame, color_hour:str, color_weekday:str):
    """
    Create the hour and weekday time distribution plot
    :param dataframe: a Weibo dataframe
    :param color_hour: the color for the hour plot
    :param color_weekday: the color for the weekday plot
    """
    fig, axis = plt.subplots(1,1, figsize=(10, 8))
    dataframe['local_time'] = dataframe.apply(
        lambda row: transform_datatime_string_to_datetime(row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe['hour'] = dataframe.apply(lambda row: row['local_time'].hour, axis=1)
    dataframe['weekday'] = dataframe.apply(lambda row: row['local_time'].weekday(), axis=1)
    fig_hour, axis_hour = plt.subplots(1, 1, figsize=(10, 8))
    fig_weekday, axis_weekday = plt.subplots(1, 1, figsize=(10, 8))
    hours_name_list = list(range(24))
    weekday_names_list = list(range(7))
    hour_count_dict = {key:0 for key in hours_name_list}
    weekday_count_dict = {key:0 for key in weekday_names_list}
    for value, sub_data in dataframe.groupby('hour'):
        hour_count_dict[value] = sub_data.shape[0]
    for value, sub_data in dataframe.groupby('weekday'):
        weekday_count_dict[value] = sub_data.shape[0]
    hours_value_list = [hour_count_dict[hour] for hour in hours_name_list]
    weekdays_value_list = [weekday_count_dict[weekday] for weekday in weekday_names_list]
    axis_hour.bar(hours_name_list, hours_value_list, color=color_hour)
    axis_weekday.bar(weekday_names_list, weekdays_value_list, color=color_weekday)
    axis_hour.set_xticks(list(range(24)))
    axis_hour.set_xlabel('Hour')
    axis_hour.set_title('Traffic-related Weibos in Shanghai by Hour')
    axis_weekday.set_xticks(list(range(7)))
    axis_weekday.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
    axis_weekday.set_xlabel('Weekday')
    axis_weekday.set_title('Traffic-related Weibos in Shanghai by Weekday')
    fig_hour.savefig(os.path.join(figures_path, 'traffic_hour.png'))
    fig_weekday.savefig(os.path.join(figures_path, 'traffic_weekday.png'))


def sentiment_distribution(dataframe:pd.DataFrame, sent_column:str, save_filename:str):
    """
    Get the overall sentiment distribution for a Weibo dataframe
    :param dataframe: a Weibo dataframe
    :param sent_column: the dataframe column which saves the sentiment analysis result
    :param save_filename: the filename of the saved figures
    """
    sentiment_result_list = []
    for index, row in dataframe.iterrows():
        sent_result = eval(row[sent_column])[0]
        sentiment_result_list.append(sent_result)
    sent_counter = Counter(sentiment_result_list)
    sent_name_list = list(sent_counter.keys())
    sent_value_list = [sent_counter[sent_label] for sent_label in sent_name_list]
    fig_sent, axis_sent = plt.subplots(1, 1, figsize=(10, 8))
    axis_sent.bar(sent_name_list, sent_value_list)
    axis_sent.set_facecolor('white')
    axis_sent.set_xticks(sent_name_list)
    axis_sent.set_xticklabels(sent_name_list, fontsize=16)
    fig_sent.savefig(os.path.join(figures_path, save_filename))


if __name__ == '__main__':
    # check_geo_data = pd.read_csv(os.path.join(weibo_data_path, 'geocoded_traffic_weibos.csv'), encoding='utf-8',
    #                              index_col=0)
    # generate_wordcloud(dataframe=check_geo_data, text_column='text', save_filename='check_geo.png')
    print('Load the dataframes for two hotspots...')
    hotspot1_data = pd.read_csv(os.path.join(shapefile_path, 'traffic_hotspot1.txt'), encoding='utf-8', index_col=0)
    hotspot2_data = pd.read_csv(os.path.join(shapefile_path, 'traffic_hotspot2.txt'), encoding='utf-8', index_col=0)
    print('Get the sentiment distribution for two hotspots...')
    sentiment_distribution(hotspot1_data, sent_column='sent_weibo', save_filename='hotspot1_sent.png')
    sentiment_distribution(hotspot2_data, sent_column='sent_weibo', save_filename='hotspot2_sent.png')
    print('Generate wordcloud for two types of traffic events (congestion & accident)...')
    generate_wordcloud_hotspots(hotspot1_data, text_column='text', accident_filename='hotspot1_accident.png',
                                congestion_filename='hotspot1_congestion.png')
    generate_wordcloud_hotspots(hotspot2_data, text_column='text', accident_filename='hotspot2_accident.png',
                                congestion_filename='hotspot2_congestion.png')

