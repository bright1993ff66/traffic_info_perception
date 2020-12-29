# For bacis
import os
import pandas as pd
import numpy as np
import pytz
from collections import Counter
from datetime import datetime, timedelta

# For plotting
from wordcloud import WordCloud
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns; sns.set(font_scale=2)

from data_paths import other_path, figures_path, hotspot_text_path, random_seed
from process_text.text_preprocessing import preprocessing_weibo
from utils import transform_datetime_string_to_datetime, count_positive_neutral_negative


# Specify the timezone in Shanghai
timezone_shanghai = pytz.timezone('Asia/Shanghai')

# Set the font size
mpl.rc('axes', labelsize=15, titlesize=20)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


def generate_wordcloud(dataframe: pd.DataFrame, save_filename: str) -> None:
    """
    Generate the wordcloud based on a Weibo dataframe
    :param dataframe: a Weibo dataframe
    :param save_filename: the filename of the saved wordcloud plot
    """
    assert 'text' in dataframe, "The dataframe should have a column named 'text' saving Weibo text content"
    font_path = os.path.join(other_path, 'SourceHanSerifK-Light.otf')
    wc = WordCloud(font_path=font_path, background_color='white', max_words=50000, max_font_size=300,
                   random_state=random_seed, collocations=False)
    if 'retweeters_text' not in dataframe:
        data_select_renamed = dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        data_select_renamed = dataframe.copy()
    text_list = list(data_select_renamed['text'])
    retweet_text_list = list(data_select_renamed['retweeters_text'])
    retweet_text_final = [text for text in retweet_text_list if text != 'no retweeters']
    combined_text_list = text_list + retweet_text_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    cleaned_text = ' '.join([preprocessing_weibo(text) for text in combined_text_without_nan])
    wc.generate_from_text(cleaned_text)
    figure, axis = plt.subplots(1, 1, dpi=200)
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    figure.savefig(os.path.join(figures_path, save_filename))


def generate_wordcloud_in_given_day(dataframe: pd.DataFrame, month: int, day: int, save_filename: str) -> None:
    """
    Generate the wordcloud based on a Weibo dataframe
    :param dataframe: a Weibo dataframe
    :param month: the month of the day that we want to generate wordcloud
    :param day: the day in the month that we want to generate wordcloud
    :param save_filename: the filename of the saved wordcloud plot
    """
    assert 'text' in dataframe, "The dataframe should have a column named 'text' saving Weibo text content"
    font_path = os.path.join(other_path, 'SourceHanSerifK-Light.otf')
    wc = WordCloud(font_path=font_path, background_color='white', max_words=50000, max_font_size=300,
                   random_state=random_seed, collocations=False)
    if 'retweeters_text' not in dataframe:
        data_renamed = dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        data_renamed = dataframe.copy()
    data_renamed['local_time'] = data_renamed.apply(lambda row: transform_datetime_string_to_datetime(
        row['local_time'], target_timezone=timezone_shanghai), axis=1)
    data_renamed['month'] = data_renamed.apply(lambda row: row['local_time'].month, axis=1)
    data_renamed['day'] = data_renamed.apply(lambda row: row['local_time'].day, axis=1)
    data_select = data_renamed.loc[(data_renamed['month'] == month) & (data_renamed['day'] == day)]
    text_list = list(data_select['text'])
    retweet_text_list = list(data_select['retweeters_text'])
    retweet_text_final = [text for text in retweet_text_list if text != 'no retweeters']
    combined_text_list = text_list + retweet_text_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    cleaned_text = ' '.join([preprocessing_weibo(text) for text in combined_text_without_nan])
    wc.generate_from_text(cleaned_text)
    figure, axis = plt.subplots(1, 1, dpi=200)
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    figure.savefig(os.path.join(figures_path, save_filename))


def generate_wordcloud_hotspots(hotspot_data: pd.DataFrame, accident_filename: str, congestion_filename: str) -> None:
    """
    Generate the wordcloud for the Weibo text found in the accident hotspot and congestion hotspot
    :param hotspot_data: Weibo data found in one traffic hotspot
    :param accident_filename: the filename of the accident wordcloud
    :param congestion_filename: the filename of the congestion wordcloud
    :return:
    """
    print('For this traffic hotspot, we have got {} weibos'.format(hotspot_data.shape[0]))
    accident_data = hotspot_data.loc[hotspot_data['traffic_ty'] == 'accident']
    congestion_data = hotspot_data.loc[hotspot_data['traffic_ty'] == 'congestion']
    generate_wordcloud(dataframe=accident_data, save_filename=accident_filename)
    generate_wordcloud(dataframe=congestion_data, save_filename=congestion_filename)


def create_day_plot(dataframe: pd.DataFrame, title: str, set_percentile: float, save_filename:str) -> None:
    """
    Create the number of traffic relevant weibos in each day plot
    :param dataframe: a Weibo dataframe
    :param title: the title of the figure
    :param set_percentile: the percentile used to set a threshold: 0 <= set_percentile <= 100
    :param save_filename: the saved figure filename in local directory
    :return: None. The figure is saved to local
    """
    dataframe_copy = dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(lambda row: transform_datetime_string_to_datetime(
        row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['month'] = dataframe_copy.apply(lambda row: row['local_time'].month, axis=1)
    dataframe_copy['day'] = dataframe_copy.apply(lambda row: row['local_time'].day, axis=1)
    repost_dataframe = dataframe_copy.loc[dataframe_copy['retweeters_text'] != 'no retweeters']
    weibo_dataframe = dataframe_copy.loc[dataframe_copy['retweeters_text'] == 'no retweeters']
    start_date, end_date = datetime(2012, 6, 1), datetime(2012, 8, 31)
    count_list, color_list = [], []
    days = mdates.drange(start_date, end_date, timedelta(days=1))
    for day_index, xtick in zip(list(range((end_date - start_date).days)), days):
        check_date = start_date + timedelta(days=day_index)
        check_month, check_day = check_date.month, check_date.day
        select_weibo_dataframe = weibo_dataframe.loc[
            (weibo_dataframe['month'] == check_month) & (weibo_dataframe['day'] == check_day)]
        select_repost_dataframe = repost_dataframe.loc[
            (repost_dataframe['month'] == check_month) & (repost_dataframe['day'] == check_day)]
        count_list.append(select_weibo_dataframe.shape[0] + select_repost_dataframe.shape[0])
    # For the days that having number of weibos bigger than a predefined the percentile
    # Highlight these days with red bar; otherwise, use the blue bar
    threshold = np.percentile(count_list, q=set_percentile)
    # print('The threshold is: {}'.format(threshold))
    for count in count_list:
        if count > threshold:
            color_list.append('red')
        else:
            color_list.append('green')

    # Plot the bars
    day_figure, day_axis = plt.subplots(1, 1, figsize=(20, 8))
    day_axis.bar(days, count_list, color=color_list)

    # xaxis setting
    day_axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    day_axis.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    day_axis.set_xlabel('Date')

    # yaxis setting
    day_axis.set_ylabel('Count')

    # Set the title and save figure
    day_axis.set_facecolor('white')
    day_axis.margins(0)
    day_axis.set_title(title)
    day_figure.autofmt_xdate()
    day_figure.savefig(os.path.join(figures_path, save_filename))


def create_hour_weekday_plot(dataframe:pd.DataFrame, color_hour:str, color_weekday:str, title_hour:str,
                             title_weekday:str, hour_save_filename:str, weekday_save_filename:str) -> None:
    """
    Create the hour and weekday time distribution plot
    :param dataframe: a Weibo dataframe
    :param color_hour: the color for the hour plot
    :param color_weekday: the color for the weekday plot
    :param title_hour: The title for the hour's figure
    :param title_weekday: The title for the weekday's figure
    :param: hour_save_filename: the name of the saved figure for hourly plot
    :param: weekday_save_filename: the name of the saved figure for weekday plot
    """
    dataframe_copy = dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(lambda row: transform_datetime_string_to_datetime(
                row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['hour'] = dataframe_copy.apply(lambda row: row['local_time'].hour, axis=1)
    dataframe_copy['weekday'] = dataframe_copy.apply(lambda row: row['local_time'].weekday(), axis=1)
    repost_dataframe = dataframe_copy.loc[dataframe_copy['retweeters_text'] != 'no retweeters']
    weibo_dataframe = dataframe_copy.loc[dataframe_copy['retweeters_text'] == 'no retweeters']

    fig_hour, axis_hour = plt.subplots(1, 1, figsize=(10, 8))
    fig_weekday, axis_weekday = plt.subplots(1, 1, figsize=(10, 8))
    hours_name_list = list(range(24))
    weekday_names_list = list(range(7))
    hour_weibo_count_dict, hour_repost_count_dict = Counter(weibo_dataframe['hour']), \
                                                    Counter(repost_dataframe['hour'])
    weekday_weibo_count_dict, weekday_repost_count_dict = Counter(weibo_dataframe['weekday']), \
                                                          Counter(repost_dataframe['weekday'])
    hours_value_list = [hour_weibo_count_dict.setdefault(hour, 0) +
                        hour_repost_count_dict.setdefault(hour, 0) for hour in hours_name_list]
    weekdays_value_list = [weekday_weibo_count_dict.setdefault(weekday, 0) +
                           weekday_repost_count_dict.setdefault(weekday, 0) for weekday in weekday_names_list]

    # create bar plot
    axis_hour.bar(hours_name_list, hours_value_list, color=color_hour, edgecolor='white')
    axis_weekday.bar(weekday_names_list, weekdays_value_list, color=color_weekday, edgecolor='white')

    # Set axis tokens
    axis_hour.set_xticks(list(range(24)))
    axis_hour.set_xlabel('Hour')
    axis_hour.set_ylabel('Count')
    axis_hour.set_title(title_hour)
    axis_weekday.set_xticks(list(range(7)))
    axis_weekday.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
    axis_weekday.set_xlabel('Weekday')
    axis_weekday.set_ylabel('Count')
    axis_weekday.set_title(title_weekday)

    # Change the plot background color and hide grids
    axis_hour.set_facecolor('white')
    axis_hour.grid(False)
    axis_weekday.set_facecolor('white')
    axis_weekday.grid(False)

    # Tight the figures and save
    fig_hour.tight_layout()
    fig_weekday.tight_layout()
    fig_hour.savefig(os.path.join(figures_path, hour_save_filename))
    fig_weekday.savefig(os.path.join(figures_path, weekday_save_filename))


def sentiment_distribution(dataframe:pd.DataFrame, sent_column:str, save_filename:str) -> None:
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


def create_hour_weekday_with_stack_sentiment(dataframe: pd.DataFrame, color_neg: str, color_neutral: str,
                                             color_pos: str, title_hour:str, title_weekday:str, hour_save_filename:str,
                                             weekday_save_filename:str) -> None:
    """
    Create the stacked sentiment barplot for both hours in a day and days in a week
    :param dataframe: the pandas dataframe saving the Weibos posted in accident or congestion hotspots
    :param color_neg: the color code for the negative sentiment bar
    :param color_neutral: the color code for the neutral sentiment bar
    :param color_pos: the color code for the positive sentiment bar
    :param title_hour: the title for the hour's figure
    :param title_weekday: the title for the weekday's figure
    :param hour_save_filename: the saved filename of the hour's figure
    :param weekday_save_filename: the saved filename of the weekday's figure
    :return: None. Figures are saved in the local directory: data_paths.figures_path
    """
    dataframe_copy = dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(lambda row: transform_datetime_string_to_datetime(
        row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['hour'] = dataframe_copy.apply(lambda row: row['local_time'].hour, axis=1)
    dataframe_copy['weekday'] = dataframe_copy.apply(lambda row: row['local_time'].weekday(), axis=1)
    fig_hour, axis_hour = plt.subplots(1, 1, figsize=(10, 8))
    fig_weekday, axis_weekday = plt.subplots(1, 1, figsize=(10, 8))
    hours_name_list = list(range(24))
    weekday_names_list = list(range(7))

    # Create the dictionaries saving the number of positive, neutral and negative Weibos across hours and days
    hour_sent_dict = {2: [], 1: [], 0: []}
    weekday_sent_dict = {2: [], 1: [], 0: []}

    # Get the count in each hour and day from the Weibo dataframe
    for hour in hours_name_list:
        dataframe_select = dataframe_copy.loc[dataframe_copy['hour'] == hour]
        # The count_positive_neutral_negative function has considered both the sentiment of Weibos and reposts
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(dataframe_select,
                                                                              repost_column='retweeters_text')
        hour_sent_dict[2].append(pos_count)
        hour_sent_dict[1].append(neutral_count)
        hour_sent_dict[0].append(neg_count)
    for weekday in weekday_names_list:
        dataframe_select = dataframe_copy.loc[dataframe_copy['weekday'] == weekday]
        # The count_positive_neutral_negative function has considered both the sentiment of Weibos and reposts
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(dataframe_select,
                                                                              repost_column='retweeters_text')
        weekday_sent_dict[2].append(pos_count)
        weekday_sent_dict[1].append(neutral_count)
        weekday_sent_dict[0].append(neg_count)
    bars_hour = np.add(hour_sent_dict[0], hour_sent_dict[1]).tolist()
    bars_weekday = np.add(weekday_sent_dict[0], weekday_sent_dict[1]).tolist()

    # Create the bar plots
    axis_hour.bar(hours_name_list, hour_sent_dict[0], color=color_neg, edgecolor='black', label='Negative')
    axis_hour.bar(hours_name_list, hour_sent_dict[1], bottom=hour_sent_dict[0], color=color_neutral, edgecolor='black',
                  label='Neutral')
    axis_hour.bar(hours_name_list, hour_sent_dict[2], bottom=bars_hour, color=color_pos, edgecolor='black',
                  label='Positive')
    axis_weekday.bar(weekday_names_list, weekday_sent_dict[0], color=color_neg, edgecolor='black', label='Negative')
    axis_weekday.bar(weekday_names_list, weekday_sent_dict[1], bottom= weekday_sent_dict[0], color=color_neutral,
                     edgecolor='black', label='Neutral')
    axis_weekday.bar(weekday_names_list, weekday_sent_dict[2], bottom=bars_weekday, color=color_pos, edgecolor='black',
                     label='Positive')

    # Set xticks
    axis_hour.set_xticks(list(range(24)))
    axis_hour.set_xlabel('Hour')
    axis_hour.set_ylabel('Count')
    axis_hour.set_title(title_hour)
    axis_weekday.set_xticks(list(range(7)))
    axis_weekday.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
    axis_weekday.set_xlabel('Weekday')
    axis_weekday.set_ylabel('Count')
    axis_weekday.set_title(title_weekday)

    # Set the background color and hide the grids
    axis_hour.set_facecolor('white')
    axis_hour.grid(False)
    axis_weekday.set_facecolor('white')
    axis_weekday.grid(False)

    # Tight the figures, create legends and save
    axis_hour.legend()
    axis_weekday.legend()
    fig_hour.tight_layout()
    fig_weekday.tight_layout()
    fig_hour.savefig(os.path.join(figures_path, hour_save_filename))
    fig_weekday.savefig(os.path.join(figures_path, weekday_save_filename))


def hotspot_day_plot(hotspot_dataframe: pd.DataFrame, title: str, set_percentile: float, save_filename: str) -> None:
    """
    Plot the number of accident or congestion Weibos posted in accident or congestion hotspots in each day
    :param hotspot_dataframe: a pandas dataframe saving the Weibo data
    :param title: the title of the plot
    :param set_percentile: the percentile used to find the abnormal days with big number of traffic Weibos
    :param save_filename: the filename of the saved figure
    :return:
    """
    # Cope with the dataframe
    hotspot_data_copy = hotspot_dataframe.copy()
    hotspot_data_copy['local_time'] = hotspot_data_copy.apply(
        lambda row: transform_datetime_string_to_datetime(row['local_time'], target_timezone=timezone_shanghai), axis=1)
    hotspot_data_copy['month'] = hotspot_data_copy.apply(lambda row: row['local_time'].month, axis=1)
    hotspot_data_copy['day'] = hotspot_data_copy.apply(lambda row: row['local_time'].day, axis=1)
    if 'retweeters_text' not in hotspot_data_copy:
        hotspot_data_renamed = hotspot_data_copy.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        hotspot_data_renamed = hotspot_data_copy.copy()
    weibo_hotspot_dataframe = hotspot_data_renamed.loc[hotspot_data_renamed['retweeters_text'] == 'no retweeters']
    repost_hotspot_dataframe = hotspot_data_renamed.loc[hotspot_data_renamed['retweeters_text'] != 'no retweeters']

    # Counting the number of original posts and reposts
    start_date, end_date = datetime(2012, 6, 1), datetime(2012, 8, 31)
    hotspot_count_list = []
    days = mdates.drange(start_date, end_date, timedelta(days=1))
    for day_index, xtick in zip(list(range((end_date - start_date).days)), days):
        check_date = start_date + timedelta(days=day_index)
        check_month, check_day = check_date.month, check_date.day
        select_weibo_hotspot_dataframe = weibo_hotspot_dataframe.loc[
            (weibo_hotspot_dataframe['month'] == check_month) & (weibo_hotspot_dataframe['day'] == check_day)]
        select_repost_hotspot_dataframe = repost_hotspot_dataframe.loc[
            (repost_hotspot_dataframe['month'] == check_month) & (repost_hotspot_dataframe['day'] == check_day)]
        hotspot_count_list.append(select_weibo_hotspot_dataframe.shape[0] + select_repost_hotspot_dataframe.shape[0])

    # For the days that having number of weibos bigger than a predefined the percentile
    # Highlight these days with red region; otherwise, use the blue bar
    hotspot_color_list = []
    threshold = np.percentile(hotspot_count_list, q=set_percentile)
    print('The threshold is: {}'.format(threshold))
    for count in hotspot_count_list:
        if count > threshold:
            hotspot_color_list.append('#FF2E52')
        else:
            hotspot_color_list.append('#FCF214')

    # Plot the bars
    hotspot_figure, hotspot_axis = plt.subplots(1, 1, figsize=(20, 8))
    bar_width = 0.4
    hotspot_axis.bar(days, hotspot_count_list, bar_width, color=hotspot_color_list)

    # xaxis setting
    hotspot_axis.xaxis_date()
    hotspot_axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    hotspot_axis.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    for label, hotspot_count in zip(hotspot_axis.xaxis.get_ticklabels(), hotspot_count_list):
        if hotspot_count > threshold:
            label.set_color('red')
            label.set_visible(True)
        else:
            label.set_visible(False)
    hotspot_axis.set_xlabel('Date')

    # yaxis setting
    hotspot_axis.set_ylabel('Count')

    # set the facecolor and disable the grid
    hotspot_axis.set_facecolor('white')
    hotspot_axis.grid(True)

    # Set the title and save figure
    hotspot_axis.margins(0)
    hotspot_axis.set_title(title)
    hotspot_figure.autofmt_xdate()
    hotspot_figure.savefig(os.path.join(figures_path, save_filename))


def hotspot_not_hotspot_day_plot(hotspot_dataframe: pd.DataFrame, not_hotspot_dataframe: pd.DataFrame, title: str,
                                 set_percentile: float, save_filename: str) -> None:
    """
    Plot the number of accident or congestion Weibos posted in accident or congestion hotspots in each day
    :param hotspot_dataframe: a pandas dataframe saving the Weibos posted in the hotspot
    :param not_hotspot_dataframe: a pandas dataframe saving the Weibos not posted in the hotspot
    :param title: the title of the plot
    :param set_percentile: the percentile used to find the abnormal days with big number of traffic Weibos
    :param save_filename: the filename of the saved figure
    :return:
    """
    hotspot_data_copy, not_hotspot_data_copy = hotspot_dataframe.copy(), not_hotspot_dataframe.copy()
    hotspot_data_copy['local_time'] = hotspot_data_copy.apply(
        lambda row: transform_datetime_string_to_datetime(row['local_time'], target_timezone=timezone_shanghai), axis=1)
    not_hotspot_data_copy['local_time'] = not_hotspot_data_copy.apply(
        lambda row: transform_datetime_string_to_datetime(row['local_time'], target_timezone=timezone_shanghai), axis=1)
    hotspot_data_copy['month'] = hotspot_data_copy.apply(lambda row: row['local_time'].month, axis=1)
    hotspot_data_copy['day'] = hotspot_data_copy.apply(lambda row: row['local_time'].day, axis=1)
    not_hotspot_data_copy['month'] = not_hotspot_data_copy.apply(lambda row: row['local_time'].month, axis=1)
    not_hotspot_data_copy['day'] = not_hotspot_data_copy.apply(lambda row: row['local_time'].day, axis=1)

    if 'retweeters_text' not in hotspot_data_copy:
        hotspot_data_renamed = hotspot_data_copy.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        hotspot_data_renamed = hotspot_data_copy.copy()

    if 'retweeters_text' not in not_hotspot_data_copy:
        not_hotspot_data_renamed = not_hotspot_data_copy.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        not_hotspot_data_renamed = not_hotspot_data_copy.copy()

    weibo_hotspot_dataframe = hotspot_data_renamed.loc[hotspot_data_renamed['retweeters_text'] == 'no retweeters']
    repost_hotspot_dataframe = hotspot_data_renamed.loc[hotspot_data_renamed['retweeters_text'] != 'no retweeters']
    weibo_not_hotspot_dataframe = not_hotspot_data_renamed.loc[
        not_hotspot_data_renamed['retweeters_text'] == 'no retweeters']
    repost_not_hotspot_dataframe = not_hotspot_data_renamed.loc[
        not_hotspot_data_renamed['retweeters_text'] != 'no retweeters']
    start_date, end_date = datetime(2012, 6, 1), datetime(2012, 8, 31)
    hotspot_count_list, not_hotspot_count_list = [], []
    days = mdates.drange(start_date, end_date, timedelta(days=1))
    for day_index, xtick in zip(list(range((end_date - start_date).days)), days):
        check_date = start_date + timedelta(days=day_index)
        check_month, check_day = check_date.month, check_date.day
        select_weibo_hotspot_dataframe = weibo_hotspot_dataframe.loc[
            (weibo_hotspot_dataframe['month'] == check_month) & (weibo_hotspot_dataframe['day'] == check_day)]
        select_repost_hotspot_dataframe = repost_hotspot_dataframe.loc[
            (repost_hotspot_dataframe['month'] == check_month) & (repost_hotspot_dataframe['day'] == check_day)]
        select_weibo_not_hotspot_dataframe = weibo_not_hotspot_dataframe.loc[
            (weibo_not_hotspot_dataframe['month'] == check_month) & (weibo_not_hotspot_dataframe['day'] == check_day)]
        select_repost_not_hotspot_dataframe = repost_not_hotspot_dataframe.loc[
            (repost_not_hotspot_dataframe['month'] == check_month) & (repost_not_hotspot_dataframe['day'] == check_day)]
        hotspot_count_list.append(select_weibo_hotspot_dataframe.shape[0] + select_repost_hotspot_dataframe.shape[0])
        not_hotspot_count_list.append(select_weibo_not_hotspot_dataframe.shape[0] +
                                      select_repost_not_hotspot_dataframe.shape[0])

    # For the days that having number of weibos bigger than a predefined the percentile
    # Highlight these days with red region; otherwise, use the blue bar
    hotspot_color_list = []
    threshold = np.percentile(hotspot_count_list, q=set_percentile)
    print('The threshold is: {}'.format(threshold))
    for count in hotspot_count_list:
        if count > threshold:
            hotspot_color_list.append('#FF2E52')
        else:
            hotspot_color_list.append('#FCF214')

    # Plot the bars
    hotspot_figure, hotspot_axis = plt.subplots(1, 1, figsize=(20, 8))
    bar_width = 0.4
    hotspot_axis.bar(days, hotspot_count_list, bar_width, color=hotspot_color_list)
    hotspot_axis.bar(days+bar_width, not_hotspot_count_list, bar_width, color=['#2DBFFC']*len(not_hotspot_count_list))

    # xaxis setting
    hotspot_axis.xaxis_date()
    hotspot_axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    hotspot_axis.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    for label, hotspot_count in zip(hotspot_axis.xaxis.get_ticklabels(), hotspot_count_list):
        if hotspot_count > threshold:
            label.set_color('red')
            label.set_visible(True)
        else:
            label.set_visible(False)
    hotspot_axis.set_xlabel('Date')

    # yaxis setting
    hotspot_axis.set_ylabel('Count')

    # set the facecolor and disable the grid
    hotspot_axis.set_facecolor('white')
    hotspot_axis.grid(True)

    # Set the title and save figure
    hotspot_axis.margins(0)
    hotspot_axis.set_title(title)
    hotspot_figure.autofmt_xdate()
    hotspot_figure.savefig(os.path.join(figures_path, save_filename))


def correlation_plot(dataframe, considered_column_list:list, save_filename:str) -> None:
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
    sns.heatmap(df_renamed.corr(), ax=axis, cmap=palette, annot=True)
    axis.set(yticks=np.arange(df_renamed.shape[1])+0.5)
    figure.savefig(os.path.join(figures_path, save_filename))


if __name__ == '__main__':
    # check_geo_data = pd.read_csv(os.path.join(weibo_data_path, 'geocoded_traffic_weibos.csv'), encoding='utf-8',
    #                              index_col=0)
    # generate_wordcloud(dataframe=check_geo_data, text_column='text', save_filename='check_geo.png')
    print('Load the dataframes for two hotspots...')
    conges_hotspot = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_cgs_hotspot_with_sent.csv'), encoding='utf-8',
                                 index_col=0)
    accident_hotspot = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_acc_hotspot_with_sent.csv'), encoding='utf-8',
                                   index_col=0)
    print('Generate wordcloud for two types of traffic events (congestion & accident)...')
    generate_wordcloud(accident_hotspot, save_filename='accident_hotspot.png')
    generate_wordcloud(conges_hotspot, save_filename='congestion_hotspot.png')
    # Check the specific days
    hotspot_days = {'acc': [(6, 22), (7, 23), (8, 28)], 'cgs': [(7, 11), (7, 20), (8, 8)]}
    for traffic_type in hotspot_days:
        if traffic_type == 'acc':
            print('Coing with the accident relevant Weibos')
            studied_time_list = hotspot_days[traffic_type]
            for day_tuple in studied_time_list:
                generate_wordcloud_in_given_day(dataframe=accident_hotspot, month=day_tuple[0], day=day_tuple[1],
                                                save_filename='acc_wordcloud_{}_{}.png'.format(day_tuple[0],
                                                                                               day_tuple[1]))
        else:
            print('Coing with the congestion relevant Weibos')
            studied_time_list = hotspot_days[traffic_type]
            for day_tuple in studied_time_list:
                generate_wordcloud_in_given_day(dataframe=conges_hotspot, month=day_tuple[0], day=day_tuple[1],
                                                save_filename='cgs_wordcloud_{}_{}.png'.format(day_tuple[0],
                                                                                               day_tuple[1]))


