# For bacis
import os
import pandas as pd
import numpy as np
from collections import Counter

# For plotting
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(font_scale=2)

from data_paths import other_path, figures_path, shapefile_path, random_seed
from process_text.text_preprocessing import preprocessing_weibo
from utils import transform_datetime_string_to_datetime
import content_analysis.time_analysis


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
    dataframe_copy = dataframe.copy()
    try:
        dataframe_copy['local_time'] = dataframe_copy.apply(
            lambda row: transform_datetime_string_to_datetime(
                row['local_time'], target_timezone=content_analysis.time_analysis.timezone_shanghai), axis=1)
    except TypeError:
        print('The datetime object has been created!')
    dataframe_copy['hour'] = dataframe_copy.apply(lambda row: row['local_time'].hour, axis=1)
    dataframe_copy['weekday'] = dataframe_copy.apply(lambda row: row['local_time'].weekday(), axis=1)
    fig_hour, axis_hour = plt.subplots(1, 1, figsize=(10, 8))
    fig_weekday, axis_weekday = plt.subplots(1, 1, figsize=(10, 8))
    hours_name_list = list(range(24))
    weekday_names_list = list(range(7))
    hour_count_dict = {key:0 for key in hours_name_list}
    weekday_count_dict = {key:0 for key in weekday_names_list}
    for value, sub_data in dataframe_copy.groupby('hour'):
        hour_count_dict[value] = sub_data.shape[0]
    for value, sub_data in dataframe_copy.groupby('weekday'):
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


def correlation_plot(dataframe, considered_column_list:list, save_filename:str):
    """
    Create the correlation plot for some columns of values in a dataframe
    All the values in the considered columns should be either int or float
    :param dataframe: the dataframe saving some values of some attributes.
    :param considered_column_list: a list containing the selected colnames
    :param save_filename: the name of the figure saving to local
    :return:
    """
    df = dataframe[considered_column_list]
    renamed_dict = {'acc_count': 'Accident Num', 'conges_count': 'Congestion Num', 'condition_count':'Condition Num',
                    'sent_index': 'Sentiment Index', 'count': 'Count'}
    df_renamed = df.rename(columns = renamed_dict)
    palette = sns.color_palette("light:b", as_cmap=True)
    for colname in df_renamed:
        assert df_renamed[colname].dtype in ['float64', 'int64'], 'The data type of column {} is not right!'.format(
            colname)
    figure, axis = plt.subplots(1, 1, figsize=(25, 18), dpi=150)
    sns.heatmap(df_renamed.corr(), ax=axis, cmap=palette)
    axis.set(yticks=np.arange(df_renamed.shape[1])+0.5)
    figure.savefig(os.path.join(figures_path, save_filename))


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

