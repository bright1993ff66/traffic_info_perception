# encoding='utf-8'
# For basics
import os
import random
import pandas as pd
import numpy as np
import pytz
from collections import Counter
from datetime import datetime, timedelta
from sklearn.metrics import auc
import geopandas as gpd
import jenkspy
from adjustText import adjust_text

# For plotting
from wordcloud import WordCloud
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

# For self-created functions
import data_paths
from data_paths import other_path, figures_path, hotspot_figures, random_seed, hotspot_figure_path
from process_text.text_preprocessing import preprocessing_weibo
from utils import transform_datetime_string_to_datetime, count_positive_neutral_negative

# Specify the timezone in Shanghai
timezone_shanghai = pytz.timezone('Asia/Shanghai')

# Set the font type
font = {'family': 'serif'}
mpl.rc('font', **font)


def plot_roc_one_setting(true_positive_rates, false_negative_rates, given_ax, label_setting,
                         select_color: str = 'blue'):
    """
    Plot the roc based on the false alarm rates and miss detection rates
    :param true_positive_rates: the true positive rates for a hotspot identification threshold
    :param false_negative_rates: the false negative rates for a hotspot identification approach
    :param given_ax: a matplotlib axis to show the created figure
    :param label_setting: the label for the line plot
    :param select_color: the color string
    :return: None. The ROC curve is saved to local directory
    """
    given_ax.plot(false_negative_rates, true_positive_rates, color=select_color, marker='.', label=label_setting)


def generate_colors(color_number: int):
    """
    Generate some random sampled colors
    :param color_number: the random color number you want to generate
    :return: a color string (if color_number = 1) or a list of colors (if color_number>1)
    """
    assert color_number >= 1, "You should at least generate one color"
    chars = '0123456789ABCDEF'
    if color_number == 1:
        return ['#'+''.join(random.sample(chars, 6)) for _ in range(color_number)][0]
    else:
        return ['#'+''.join(random.sample(chars, 6)) for _ in range(color_number)]


def compare_hotspot_roc(hotspot_compare_dataframe: pd.DataFrame, considered_traffic_type: str):

    """
    Generate the roc plot for hotspot comparison
    :param hotspot_compare_dataframe: a pandas dataframe saving the hotspot compare result
    :param considered_traffic_type: the considered traffic event type, either 'acc' or 'cgs'
    :return: area under the curve for each hotspot setting based on traffic-relevant Weibos and
    actual traffic records
    """

    assert considered_traffic_type in ['acc', 'cgs'], "The traffic type should be either 'acc' or 'cgs'"

    figure_actual, axis_actual = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
    figure_weibo, axis_weibo = plt.subplots(1, 1, figsize=(10, 8), dpi=300)

    # Generate colors for each hotspot identification setting
    auc_dict_actual, auc_dict_weibo = {}, {}
    # number_of_hotspot_settings = len(set(hotspot_compare_dataframe['setting']))
    hotspot_settings = list(set(hotspot_compare_dataframe['setting']))
    color_codes = ['#FFAF37', '#6F55FF', '#44FFA2', '#C9FF40', '#FF50CD', '#FF4D44']
    color_dict = {setting: color for setting, color in zip(hotspot_settings, color_codes)}

    # Plot the roc curve for each hotspot setting - based on actual traffic records
    for setting, hotspot_dataframe in hotspot_compare_dataframe.groupby('setting'):
        tpr_rates = list(hotspot_dataframe['TPR_actual'])
        fpr_rates = list(hotspot_dataframe['FPR_actual'])
        axis_actual.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.8)  # Draw the baseline
        random_color = color_dict[setting]
        auc_dict_actual[setting] = auc(fpr_rates, tpr_rates)
        if 'with_sent' in setting:
            hotspot_label_setting = "Unit Size: {} meter; Consider Sentiment: True".format(setting.split('_')[2])
        else:
            hotspot_label_setting = "Unit Size: {} meter; Consider Sentiment: False".format(setting.split('_')[2])
        plot_roc_one_setting(true_positive_rates=tpr_rates, false_negative_rates=fpr_rates, given_ax=axis_actual,
                             label_setting=hotspot_label_setting, select_color=random_color)

    # Plot the roc curve for each hotspot setting - based on traffic-relevant Weibos
    for setting, hotspot_dataframe in hotspot_compare_dataframe.groupby('setting'):
        tpr_rates = list(hotspot_dataframe['TPR_weibo'])
        fpr_rates = list(hotspot_dataframe['FPR_weibo'])
        axis_weibo.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.8)  # Draw the baseline
        random_color = color_dict[setting]
        auc_dict_weibo[setting] = auc(fpr_rates, tpr_rates)
        if 'with_sent' in setting:
            hotspot_label_setting = "Unit Size: {} meter; Consider Sentiment: True".format(setting.split('_')[2])
        else:
            hotspot_label_setting = "Unit Size: {} meter; Consider Sentiment: False".format(setting.split('_')[2])
        plot_roc_one_setting(true_positive_rates=tpr_rates, false_negative_rates=fpr_rates, given_ax=axis_weibo,
                             label_setting=hotspot_label_setting, select_color=random_color)

    axis_actual.legend(fontsize=13)
    axis_weibo.legend(fontsize=13)
    axis_actual.spines['right'].set_visible(False)
    axis_actual.spines['top'].set_visible(False)
    axis_weibo.spines['right'].set_visible(False)
    axis_weibo.spines['top'].set_visible(False)

    axis_actual.set_xlabel('False Positive Rates\n(False Alarm Rates)', size=25)
    axis_actual.set_ylabel('True Positive Rates\n(1 - Miss Detection Rates)', size=25)
    axis_weibo.set_xlabel('False Positive Rates\n(False Alarm Rates)', size=25)
    axis_weibo.set_ylabel('True Positive Rates\n(1 - Miss Detection Rates)', size=25)
    axis_actual.xaxis.set_tick_params(labelsize=20)
    axis_actual.yaxis.set_tick_params(labelsize=20)
    axis_weibo.xaxis.set_tick_params(labelsize=20)
    axis_weibo.yaxis.set_tick_params(labelsize=20)

    if considered_traffic_type == 'acc':
        axis_actual.set_title('ROC Curve based on Actual Traffic Accidents', size=25)
        axis_weibo.set_title('ROC Curve based on Accident-relevant Weibos', size=25)
    else:
        axis_actual.set_title('ROC Curve based on Actual Traffic Congestions', size=25)
        axis_weibo.set_title('ROC Curve based on Congestion-relevant Weibos', size=25)

    figure_actual.savefig(os.path.join(figures_path, '{}_actual_compare.png'.format(considered_traffic_type)),
                          bbox_inches='tight')
    figure_weibo.savefig(os.path.join(figures_path, '{}_weibo_compare.png'.format(considered_traffic_type)),
                         bbox_inches='tight')
    return auc_dict_actual, auc_dict_weibo


def generate_wordcloud(dataframe: pd.DataFrame, save_filename: str, save_path: str = data_paths.figures_path) -> None:
    """
    Generate the wordcloud based on a Weibo dataframe
    :param save_path: the path to save the created figure
    :param dataframe: a Weibo dataframe
    :param save_filename: the filename of the saved wordcloud plot
    """
    assert 'text' in dataframe, "The dataframe should have a column named 'text' saving Weibo text content"
    font_path = os.path.join(other_path, 'SourceHanSerifK-Light.otf')  # Load the font
    wc = WordCloud(font_path=font_path, background_color='white', max_words=50000, max_font_size=300,
                   random_state=random_seed, collocations=False)
    if 'retweeters_text' not in dataframe:  # Change the column names produced by ArcGIS
        data_select_renamed = dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        data_select_renamed = dataframe.copy()
    text_list = list(data_select_renamed['text'])
    retweet_text_list = list(data_select_renamed['retweeters_text'])
    retweet_text_final = [text for text in retweet_text_list if text != 'no retweeters']
    combined_text_list = text_list + retweet_text_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    cleaned_text = ' '.join([preprocessing_weibo(text, return_word_list=False) for text in combined_text_without_nan])
    wc.generate_from_text(cleaned_text)
    figure, axis = plt.subplots(1, 1, dpi=200)
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    figure.savefig(os.path.join(save_path, save_filename))


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
    cleaned_text = ' '.join([preprocessing_weibo(text, return_word_list=False) for text in combined_text_without_nan])
    wc.generate_from_text(cleaned_text)
    figure, axis = plt.subplots(1, 1, dpi=300)
    plt.imshow(wc)
    plt.axis('off')
    plt.tight_layout(pad=0)
    figure.savefig(os.path.join(hotspot_figures, save_filename))


def generate_wordcloud_in_given_month(dataframe: pd.DataFrame, month: int, save_filename: str,
                                      save_path: str = hotspot_figures) -> None:
    """
    Generate the wordcloud based on a Weibo dataframe
    :param dataframe: a Weibo dataframe
    :param month: the month of the day that we want to generate wordcloud
    :param save_filename: the filename of the saved wordcloud plot
    :param save_path: the path used to save the wordcloud
    """
    assert 'text' in dataframe, "The dataframe should have a column named 'text' saving Weibo text content"
    font_path = os.path.join(other_path, 'SourceHanSerifK-Light.otf')

    if 'retweeters_text' not in dataframe:
        data_renamed = dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        data_renamed = dataframe.copy()

    # Create the text for wordcloud
    data_renamed['local_time'] = data_renamed.apply(lambda row: transform_datetime_string_to_datetime(
        row['local_time'], target_timezone=timezone_shanghai), axis=1)
    data_renamed['month'] = data_renamed.apply(lambda row: row['local_time'].month, axis=1)
    data_select = data_renamed.loc[(data_renamed['month'] == month)]
    text_list = list(data_select['text'])
    retweet_text_list = list(data_select['retweeters_text'])
    retweet_text_final = [text for text in retweet_text_list if text != 'no retweeters']
    combined_text_list = text_list + retweet_text_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    cleaned_text = ' '.join([preprocessing_weibo(text, return_word_list=False) for text in combined_text_without_nan])

    # Generate wordcloud figure
    wc = WordCloud(font_path=font_path, background_color='white', max_words=50000, max_font_size=300,
                   random_state=random_seed, collocations=False)
    try:
        wc.generate_from_text(cleaned_text)
        figure, axis = plt.subplots(1, 1, dpi=300)
        plt.imshow(wc)
        plt.axis('off')
        figure.savefig(os.path.join(save_path, save_filename), bbox_inches='tight')
    except ValueError:
        print('At least one word is needed. Skip this setting')


def generate_wordcloud_hotspots(hotspot_data: pd.DataFrame, accident_filename: str, congestion_filename: str) -> None:
    """
    Generate the wordcloud for the Weibo text found in the accident hotspot and congestion hotspot
    :param hotspot_data: Weibo data found in one traffic hotspot
    :param accident_filename: the filename of the accident wordcloud
    :param congestion_filename: the filename of the congestion wordcloud
    :return: None. The generated word clouds are saved to the local directory
    """
    assert 'traffic_ty' in hotspot_data, "The dataframe should have one column saving traffic event type"
    print('For this traffic hotspot, we have got {} weibos'.format(hotspot_data.shape[0]))
    accident_data = hotspot_data.loc[hotspot_data['traffic_ty'] == 'accident']
    congestion_data = hotspot_data.loc[hotspot_data['traffic_ty'] == 'congestion']
    generate_wordcloud(dataframe=accident_data, save_filename=accident_filename)
    generate_wordcloud(dataframe=congestion_data, save_filename=congestion_filename)


def create_day_plot(dataframe: pd.DataFrame, title: str, set_percentile: float, save_filename: str,
                    save_path: str = data_paths.figures_path) -> None:
    """
    Create the number of traffic relevant weibos in each day plot
    :param save_path: the path used to save the created figure
    :param dataframe: a Weibo dataframe
    :param title: the title of the figure
    :param set_percentile: the percentile used to set a threshold: 0 <= set_percentile <= 100
    :param save_filename: the saved figure filename in local directory
    :return: None. The figure is saved to local
    """
    if 'retweeters_text' not in dataframe:
        dataframe_copy = dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        dataframe_copy = dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(lambda row: transform_datetime_string_to_datetime(
        row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['month'] = dataframe_copy.apply(lambda row: row['local_time'].month, axis=1)
    dataframe_copy['day'] = dataframe_copy.apply(lambda row: row['local_time'].day, axis=1)
    start_date, end_date = datetime(2012, 6, 1), datetime(2012, 9, 1)
    count_list, color_list = [], []
    days = mdates.drange(start_date, end_date, timedelta(days=1))
    for day_index, xtick in zip(list(range((end_date - start_date).days)), days):
        check_date = start_date + timedelta(days=day_index)
        check_month, check_day = check_date.month, check_date.day
        dataframe_select = dataframe_copy.loc[
            (dataframe_copy['month'] == check_month) & (dataframe_copy['day'] == check_day)]
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(dataframe_select,
                                                                              repost_column='retweeters_text')
        count_list.append(pos_count + neutral_count + neg_count)
    # For the days that having number of weibos bigger than a predefined the percentile
    # Highlight these days with red bar; otherwise, use the blue bar
    threshold = np.percentile(count_list, q=set_percentile)
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
    day_axis.set_xlabel('Date', size=20)

    # yaxis setting
    day_axis.set_ylabel('# of Traffic-Relevant Weibos', size=20)
    day_axis.yaxis.set_tick_params(labelsize=20)

    # Set the top and right axis to be invisible:
    # https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
    day_axis.spines['right'].set_visible(False)
    day_axis.spines['top'].set_visible(False)

    # Set the title and save figure
    day_axis.set_facecolor('white')
    day_axis.margins(0)
    day_axis.set_title(title)
    day_figure.autofmt_xdate()
    day_figure.savefig(os.path.join(save_path, save_filename), bbox_inches='tight')


def create_hour_weekday_plot(dataframe: pd.DataFrame, color_hour: str, color_weekday: str, title_hour: str,
                             title_weekday: str, hour_save_filename: str, weekday_save_filename: str) -> None:
    """
    Create the hour and weekday time distribution plot
    Take the number of both Weibo and repost (if available) into account
    :param dataframe: a Weibo dataframe
    :param color_hour: the color for the hour plot
    :param color_weekday: the color for the weekday plot
    :param title_hour: The title for the hour's figure
    :param title_weekday: The title for the weekday's figure
    :param hour_save_filename: the name of the saved figure for hourly plot
    :param weekday_save_filename: the name of the saved figure for weekday plot
    """
    # Process the time of Weibos
    if 'retweeters_text' not in dataframe:
        dataframe_copy = dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        dataframe_copy = dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(lambda row: transform_datetime_string_to_datetime(
        row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['hour'] = dataframe_copy.apply(lambda row: row['local_time'].hour, axis=1)
    dataframe_copy['weekday'] = dataframe_copy.apply(lambda row: row['local_time'].weekday(), axis=1)

    fig_hour, axis_hour = plt.subplots(1, 1, figsize=(10, 8))
    axis_hour_neg = axis_hour.twinx()
    fig_weekday, axis_weekday = plt.subplots(1, 1, figsize=(10, 8))
    axis_weekday_neg = axis_weekday.twinx()
    hours_name_list = list(range(24))
    weekday_names_list = list(range(7))  # 0 means Monday and 6 means Sunday

    # Each Weibo has its hour and weekday information
    hours_value_list, hours_neg_percent_list = [], []
    for hour in hours_name_list:
        dataframe_select = dataframe_copy.loc[dataframe_copy['hour'] == hour]
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(dataframe_select, 'retweeters_text')
        hours_value_list.append(pos_count + neutral_count + neg_count)
        if pos_count + neutral_count + neg_count == 0:
            hours_neg_percent_list.append(0)
        else:
            hours_neg_percent_list.append(np.round(neg_count/(pos_count + neutral_count + neg_count), 2))
    weekdays_value_list, weekdays_neg_percent_list = [], []
    for weekday in weekday_names_list:
        dataframe_select = dataframe_copy.loc[dataframe_copy['weekday'] == weekday]
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(dataframe_select, 'retweeters_text')
        weekdays_value_list.append(pos_count + neutral_count + neg_count)
        weekdays_neg_percent_list.append(np.round(neg_count/(pos_count + neutral_count + neg_count), 2))

    # create bar plot
    axis_hour.bar(hours_name_list, hours_value_list, color=color_hour, edgecolor='white')
    axis_hour_neg.plot(hours_name_list, hours_neg_percent_list, color='black', linestyle='dashed', alpha=0.5)
    for hour_name, hour_val, hours_neg_percent in zip(hours_name_list, hours_value_list, hours_neg_percent_list):
        axis_hour.text(hour_name-0.3, hour_val+3, '{}'.format(hours_neg_percent))
    axis_weekday.bar(weekday_names_list, weekdays_value_list, color=color_weekday, edgecolor='white')
    axis_weekday_neg.plot(weekday_names_list, weekdays_neg_percent_list, color='black', linestyle='dashed', alpha=0.5)
    for weekday_name, weekday_val, weekday_neg_percent in zip(weekday_names_list, weekdays_value_list,
                                                              weekdays_neg_percent_list):
        axis_weekday.text(weekday_name-0.2, weekday_val+5, '{}'.format(weekday_neg_percent))

    # Set axis tokens
    axis_hour.set_xticks(list(range(24)))
    axis_hour.set_xlabel('Hour')
    axis_hour.set_ylabel('Count')
    axis_hour_neg.set_ylabel('% of Negative Weibos')
    axis_hour.set_title(title_hour)
    axis_weekday.set_xticks(list(range(7)))
    axis_weekday.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
    axis_weekday.set_xlabel('Weekday')
    axis_weekday.set_ylabel('Count')
    axis_weekday_neg.set_ylabel('% of Negative Weibos')
    axis_weekday.set_title(title_weekday)

    # Change the plot background color and hide grids
    axis_hour.set_facecolor('white')
    axis_hour.grid(False)
    axis_weekday.set_facecolor('white')
    axis_weekday.grid(False)

    # Set the top and right axis to be invisible:
    # https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
    axis_hour.spines['right'].set_visible(False)
    axis_hour.spines['top'].set_visible(False)
    axis_hour_neg.spines['top'].set_visible(False)
    axis_weekday.spines['right'].set_visible(False)
    axis_weekday.spines['top'].set_visible(False)
    axis_weekday_neg.spines['top'].set_visible(False)

    # Set the ylim
    axis_hour_neg.set_ylim(0, 1)
    axis_weekday_neg.set_ylim(0, 1)

    # Tight the figures and save
    fig_hour.tight_layout()
    fig_weekday.tight_layout()
    fig_hour.savefig(os.path.join(figures_path, hour_save_filename))
    fig_weekday.savefig(os.path.join(figures_path, weekday_save_filename))


def sentiment_distribution(dataframe: pd.DataFrame, sent_column: str, save_filename: str) -> None:
    """
    Get the overall sentiment distribution for a Weibo dataframe
    :param dataframe: a Weibo dataframe
    :param sent_column: the dataframe column which saves the sentiment analysis result
    :param save_filename: the filename of the saved figures
    """
    # Create the sentiment distribution
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
                                             color_pos: str,
                                             hour_save_filename: str,
                                             weekday_save_filename: str) -> None:
    """
    Create the stacked sentiment barplot for both hours in a day and days in a week
    :param dataframe: the pandas dataframe saving the Weibos posted in accident or congestion hotspots
    :param color_neg: the color code for the negative sentiment bar
    :param color_neutral: the color code for the neutral sentiment bar
    :param color_pos: the color code for the positive sentiment bar
    :param hour_save_filename: the saved filename of the hour's figure
    :param weekday_save_filename: the saved filename of the weekday's figure
    :return: None. Figures are saved in the local directory: data_paths.figures_path
    """
    # Process the Weibos' time
    if 'retweeters_text' not in dataframe:
        dataframe_copy = dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        dataframe_copy = dataframe.copy()

    dataframe_copy['local_time'] = dataframe_copy.apply(lambda row: transform_datetime_string_to_datetime(
        row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['hour'] = dataframe_copy.apply(lambda row: row['local_time'].hour, axis=1)
    dataframe_copy['weekday'] = dataframe_copy.apply(lambda row: row['local_time'].weekday(), axis=1)

    # Create the figure & axis objects
    fig_hour, axis_hour = plt.subplots(1, 1, figsize=(16, 12.8))
    axis_hour_neg = axis_hour.twinx()
    fig_weekday, axis_weekday = plt.subplots(1, 1, figsize=(10, 8))
    axis_weekday_neg = axis_weekday.twinx()

    # Create the dictionaries saving the number of positive, neutral and negative Weibos across hours and days
    hours_name_list = list(range(24))
    weekday_names_list = list(range(7))
    hour_sent_dict = {2: [], 1: [], 0: []}
    weekday_sent_dict = {2: [], 1: [], 0: []}

    # Get the count in each hour and day from the Weibo dataframe
    hours_neg_percent_list = []
    for hour in hours_name_list:
        dataframe_select = dataframe_copy.loc[dataframe_copy['hour'] == hour]
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(dataframe_select,
                                                                              repost_column='retweeters_text')
        hour_sent_dict[2].append(pos_count)
        hour_sent_dict[1].append(neutral_count)
        hour_sent_dict[0].append(neg_count)
        if pos_count + neutral_count + neg_count == 0:
            hours_neg_percent_list.append(0)
        else:
            hours_neg_percent_list.append(neg_count/(pos_count + neutral_count + neg_count))
    weekdays_neg_percent_list = []
    for weekday in weekday_names_list:
        dataframe_select = dataframe_copy.loc[dataframe_copy['weekday'] == weekday]
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(dataframe_select,
                                                                              repost_column='retweeters_text')
        weekday_sent_dict[2].append(pos_count)
        weekday_sent_dict[1].append(neutral_count)
        weekday_sent_dict[0].append(neg_count)
        if pos_count + neutral_count + neg_count == 0:
            weekdays_neg_percent_list.append(0)
        else:
            weekdays_neg_percent_list.append(neg_count/(pos_count + neutral_count + neg_count))
    bars_hour = np.add(hour_sent_dict[0], hour_sent_dict[1]).tolist()
    bars_weekday = np.add(weekday_sent_dict[0], weekday_sent_dict[1]).tolist()

    # Create the bar plots and negative percentage line plots
    axis_hour.bar(hours_name_list, hour_sent_dict[0], color=color_neg, edgecolor='black', label='Negative')
    axis_hour.bar(hours_name_list, hour_sent_dict[1], bottom=hour_sent_dict[0], color=color_neutral, edgecolor='black',
                  label='Neutral')
    axis_hour.bar(hours_name_list, hour_sent_dict[2], bottom=bars_hour, color=color_pos, edgecolor='black',
                  label='Positive')
    axis_hour_neg.plot(hours_name_list, hours_neg_percent_list, color='black', linestyle='dashed', alpha=0.6)
    axis_weekday.bar(weekday_names_list, weekday_sent_dict[0], color=color_neg, edgecolor='black', label='Negative')
    axis_weekday.bar(weekday_names_list, weekday_sent_dict[1], bottom=weekday_sent_dict[0], color=color_neutral,
                     edgecolor='black', label='Neutral')
    axis_weekday.bar(weekday_names_list, weekday_sent_dict[2], bottom=bars_weekday, color=color_pos, edgecolor='black',
                     label='Positive')
    axis_weekday_neg.plot(weekday_names_list, weekdays_neg_percent_list, color='black', linestyle='dashed', alpha=0.6)

    # Set xticks, yticks, xlabel, and ylabel
    axis_hour.set_xticks(list(range(24)))
    axis_hour.set_xticklabels(list(range(24)), size=30)
    axis_hour.set_xlabel('Hour', size=35)
    axis_hour.yaxis.set_tick_params(labelsize=30)
    axis_hour.set_ylabel('# of Weibos', size=35)
    axis_hour_neg.set_ylabel('% of Negative Weibos', size=35)
    axis_hour_neg.yaxis.set_tick_params(labelsize=35)

    axis_weekday.set_xticks(list(range(7)))
    axis_weekday.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'], size=20)
    axis_weekday.set_xlabel('Weekday', size=20)
    axis_weekday.yaxis.set_tick_params(labelsize=20)
    axis_weekday.set_ylabel('# of Weibos', size=20)
    axis_weekday_neg.set_ylabel('% of Negative Weibos', size=20)
    axis_weekday_neg.yaxis.set_tick_params(labelsize=22)

    # Set the background color and hide the grids
    axis_hour.set_facecolor('white')
    axis_hour.grid(False)
    axis_weekday.set_facecolor('white')
    axis_weekday.grid(False)

    # Set the top and right axis to be invisible:
    # https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
    axis_hour.spines['right'].set_visible(False)
    axis_hour.spines['top'].set_visible(False)
    axis_hour_neg.spines['top'].set_visible(False)
    axis_weekday.spines['right'].set_visible(False)
    axis_weekday.spines['top'].set_visible(False)
    axis_weekday_neg.spines['top'].set_visible(False)

    # Set the ylim
    axis_hour_neg.set_ylim(0, 1)
    axis_weekday_neg.set_ylim(0, 1)

    # Tight the figures, create legends and save
    axis_hour.legend(fontsize=20)
    axis_weekday.legend(fontsize=20)
    fig_hour.tight_layout()
    fig_weekday.tight_layout()
    fig_hour.savefig(os.path.join(figures_path, hour_save_filename))
    fig_weekday.savefig(os.path.join(figures_path, weekday_save_filename))


def hotspot_day_plot(hotspot_dataframe: pd.DataFrame, title: str, set_percentile: float, color_pos: str,
                     color_neutral: str, color_neg: str, save_filename: str, plot_threshold: bool = True,
                     save_path: str = figures_path) -> None:
    """
    Plot the number of accident or congestion Weibos posted in accident or congestion hotspots in each day
    :param hotspot_dataframe: a pandas dataframe saving the Weibo data
    :param title: the title of the plot
    :param set_percentile: the percentile used to find the abnormal days with big number of traffic Weibos
    :param color_pos: the color code for the positive sentiment bar
    :param color_neutral: the color code for the neutral sentiment bar
    :param color_neg: the color code for the negative sentiment bar
    :param save_filename: the filename of the saved figure
    :param plot_threshold: plot the 97.5% threshold or not
    :param save_path: the path used to save the figures
    :return: None. The figures are saved in the local directory
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

    # Counting the number of Weibos
    start_date, end_date = datetime(2012, 6, 1), datetime(2012, 9, 1)
    days = mdates.drange(start_date, end_date, timedelta(days=1))
    hotspot_count_list = []
    hotspot_day_count_dict = {2: [], 1: [], 0: []}
    for day_index, xtick in zip(list(range((end_date - start_date).days)), days):
        check_date = start_date + timedelta(days=day_index)
        check_month, check_day = check_date.month, check_date.day
        select_dataframe = hotspot_data_renamed.loc[
            (hotspot_data_renamed['month'] == check_month) & (hotspot_data_renamed['day'] == check_day)]
        # if (check_month == 7) & (check_day == 11):
        #     select_dataframe.to_csv(os.path.join(other_path, 'check_7_11_cgs.csv'), encoding='utf-8')
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(select_dataframe,
                                                                              repost_column='retweeters_text')
        hotspot_count_list.append(pos_count + neutral_count + neg_count)
        hotspot_day_count_dict[2].append(pos_count)
        hotspot_day_count_dict[1].append(neutral_count)
        hotspot_day_count_dict[0].append(neg_count)

    # Plot the bars
    hotspot_figure, hotspot_axis = plt.subplots(1, 1, figsize=(22, 8))
    bars_day = np.add(hotspot_day_count_dict[0], hotspot_day_count_dict[1]).tolist()
    hotspot_axis.bar(days, hotspot_day_count_dict[0], color=color_neg, edgecolor='black', label='Negative')
    hotspot_axis.bar(days, hotspot_day_count_dict[1], bottom=hotspot_day_count_dict[0], color=color_neutral,
                     edgecolor='black', label='Neutral')
    hotspot_axis.bar(days, hotspot_day_count_dict[2], bottom=bars_day, color=color_pos, edgecolor='black',
                     label='Positive')

    # xaxis setting
    hotspot_axis.xaxis_date()
    hotspot_axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    if plot_threshold:
        hotspot_axis.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        threshold = np.percentile(hotspot_count_list, set_percentile)
        hotspot_axis.axhline(y=threshold, color='black', linestyle='--')
        hotspot_axis.text(days[5], threshold + 3, "97.5% Percentile: {}".format(round(threshold, 0)), fontsize=15)
        # Only show the xticks when the number of Weibos exceeds a threshold (97.5% percentile)
        for label, hotspot_count in zip(hotspot_axis.xaxis.get_ticklabels(), hotspot_count_list):
            if hotspot_count > threshold:
                label.set_color('red')
                label.set_visible(True)
            else:
                label.set_visible(False)
    else:
        hotspot_axis.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    hotspot_axis.xaxis.set_tick_params(labelsize=20)
    hotspot_axis.set_xlabel('Date', size=20)

    # yaxis setting
    # if 'acc' in save_filename:
    #     hotspot_axis.set_ylim(0, 100)
    # else:
    #     hotspot_axis.set_ylim(0, 100)
    hotspot_axis.yaxis.set_tick_params(labelsize=20)
    hotspot_axis.set_ylabel('Count', size=20)

    # set the facecolor and disable the grid
    hotspot_axis.set_facecolor('white')
    hotspot_axis.grid(False)

    # Set the top and right axis to be invisible:
    # https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
    hotspot_axis.spines['right'].set_visible(False)
    hotspot_axis.spines['top'].set_visible(False)

    # Set the title and save figure
    hotspot_axis.legend(fontsize=20)
    hotspot_axis.set_title(title, size=25)
    hotspot_axis.margins(0)
    hotspot_figure.autofmt_xdate()
    hotspot_figure.savefig(os.path.join(save_path, save_filename), bbox_inches='tight')


def hotspot_actual_day_plot(hotspot_dataframe: pd.DataFrame, actual_dataframe: pd.DataFrame, title: str,
                            set_percentile: float, color_neg: str, color_neutral: str,
                            color_pos: str, save_filename: str) -> None:
    """
    Plot the number of accident or congestion Weibos posted in accident or congestion hotspots in each day
    :param hotspot_dataframe: a pandas dataframe saving the Weibo data
    :param actual_dataframe: a pandas dataframe saving the actual traffic records
    :param title: the title of the plot
    :param set_percentile: the percentile used to find the abnormal days with big number of traffic Weibos
    :param color_pos: the color code for the positive sentiment bar
    :param color_neutral: the color code for the neutral sentiment bar
    :param color_neg: the color code for the negative sentiment bar
    :param save_filename: the filename of the saved figure
    :return: None. The figures are saved in the local directory
    """
    # Cope with the dataframe
    assert 'acc' in save_filename or 'cgs' in save_filename, "The filename should contain either 'acc' or 'cgs'"
    hotspot_data_copy = hotspot_dataframe.copy()
    hotspot_data_copy['local_time'] = hotspot_data_copy.apply(
        lambda row: transform_datetime_string_to_datetime(row['local_time'], target_timezone=timezone_shanghai), axis=1)
    hotspot_data_copy['month'] = hotspot_data_copy.apply(lambda row: row['local_time'].month, axis=1)
    hotspot_data_copy['day'] = hotspot_data_copy.apply(lambda row: row['local_time'].day, axis=1)
    actual_data_copy = actual_dataframe.copy()
    actual_data_copy['time'] = pd.to_datetime(actual_data_copy.time)
    actual_data_copy['month'] = actual_data_copy.apply(lambda row: row['time'].month, axis=1)
    actual_data_copy['day'] = actual_data_copy.apply(lambda row: row['time'].day, axis=1)
    if 'retweeters_text' not in hotspot_data_copy:
        hotspot_data_renamed = hotspot_data_copy.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        hotspot_data_renamed = hotspot_data_copy.copy()

    # Counting the number of Weibos
    start_date, end_date = datetime(2012, 6, 1), datetime(2012, 9, 1)
    days = mdates.drange(start_date, end_date, timedelta(days=1))
    hotspot_count_list, actual_count_list = [], []
    hotspot_day_count_dict = {2: [], 1: [], 0: []}
    for day_index, xtick in zip(list(range((end_date - start_date).days)), days):
        check_date = start_date + timedelta(days=day_index)
        check_month, check_day = check_date.month, check_date.day
        select_weibo_dataframe = hotspot_data_renamed.loc[
            (hotspot_data_renamed['month'] == check_month) & (hotspot_data_renamed['day'] == check_day)]
        select_actual_dataframe = actual_data_copy.loc[
            (actual_data_copy['month'] == check_month) & (actual_data_copy['day'] == check_day)]
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(select_weibo_dataframe,
                                                                              repost_column='retweeters_text')
        hotspot_count_list.append(pos_count + neutral_count + neg_count)
        actual_count_list.append(select_actual_dataframe.shape[0])
        hotspot_day_count_dict[2].append(pos_count)
        hotspot_day_count_dict[1].append(neutral_count)
        hotspot_day_count_dict[0].append(neg_count)

    # Plot the bars
    hotspot_figure, hotspot_axis = plt.subplots(1, 1, figsize=(22, 8))
    bars_day = np.add(hotspot_day_count_dict[0], hotspot_day_count_dict[1]).tolist()
    hotspot_axis.bar(days, hotspot_day_count_dict[0], color=color_neg, edgecolor='black',
                     label='Negative')
    hotspot_axis.bar(days, hotspot_day_count_dict[1], bottom=hotspot_day_count_dict[0],
                     color=color_neutral, edgecolor='black', label='Neutral')
    hotspot_axis.bar(days, hotspot_day_count_dict[2], bottom=bars_day, color=color_pos,
                     edgecolor='black', label='Positive')
    if 'acc' in save_filename:
        hotspot_axis.plot(days, actual_count_list, color="black", linestyle='dashdot',
                          label='Actual Traffic Accidents')
    else:
        hotspot_axis.plot(days, actual_count_list, color="black", linestyle='dashdot',
                          label='Actual Traffic Congestions')

    # xaxis setting
    hotspot_axis.xaxis_date()
    hotspot_axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    hotspot_axis.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    threshold = np.percentile(hotspot_count_list, set_percentile)
    # hotspot_axis.axhline(y=threshold, color='black', linestyle='--')
    # hotspot_axis.text(days[5], threshold+5, "97.5% Percentile: {}".format(round(threshold, 0)), fontsize=15)
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
    hotspot_axis.grid(False)

    # Set the top and right axis to be invisible:
    # https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
    hotspot_axis.spines['right'].set_visible(False)
    hotspot_axis.spines['top'].set_visible(False)

    # Set the title and save figure
    hotspot_axis.legend()
    hotspot_axis.margins(0)
    hotspot_axis.set_title(title)
    hotspot_figure.autofmt_xdate()
    hotspot_figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def draw_kde_densities(density_values, consider_sentiment, threshold_value, save_filename,
                       save_path=hotspot_figure_path, spatial_unit_size=260):
    """
    The histogram of density values outputed by ArcGIS
    :param density_values: the values saving the density of spatial units
    :param consider_sentiment: consider sentiment or not in the kernel density computation or not
    :param threshold_value: the threshold value to find the hotspot areas
    :param save_path: the path used to save the created figure
    :param save_filename: the name of the saved figure
    :param spatial_unit_size: the considered spatial unit size
    :return: None. The histogram plot is saved to local directory
    """
    assert 'acc' in save_filename or 'cgs' in save_filename, 'The saved figure name should contain traffic type: ' \
                                                             'acc for accident and cgs for congestion'
    # Create a figure
    figure, axis_kde = plt.subplots(1, 1, figsize=(12, 8), dpi=300)

    # Draw the histogram
    if 'acc' in save_filename:  # If we are studying accident
        print('Creating the density histogram for Weibo accidents...')
        if consider_sentiment:
            axis_kde.set_title(
                'Histogram of KDE Values to Detect Accident Hotspots\n(Consider Sentiment; {}-meter Spatial Unit)'.format(
                    spatial_unit_size), size=22)
        else:
            axis_kde.set_title(
                'Histogram of KDE Values to Detect Accident Hotspots\n(Not Consider Sentiment; {}-meter Spatial Unit)'.format(
                    spatial_unit_size), size=22)
    else:  # If we are studying congestion
        print('Creating the density histogram for Weibo congestions...')
        if consider_sentiment:
            axis_kde.set_title(
                'Histogram of KDE Values to Detect Congestion Hotspots\n(Consider Sentiment; {}-meter Spatial Unit)'.format(
                    spatial_unit_size), size=22)
        else:
            axis_kde.set_title(
                'Histogram of KDE Values to Detect Congestion Hotspots\n(Not Consider Sentiment; {}-meter Spatial Unit)'.format(
                    spatial_unit_size), size=22)

    y_vals, x_vals, _ = axis_kde.hist(density_values, bins=50)

    # Compute the threshold value
    print('The threshold value is set to: {}'.format(threshold_value))
    # Draw the threshold line to determine the hotspot
    axis_kde.axvline(threshold_value, linestyle='--', color='red')

    if spatial_unit_size > 260:
        axis_kde.text(threshold_value + 0.15, 60000, 'Hotspot Threshold: \n{}'.format(round(threshold_value, 4)),
                      color='red', size=15)
    else:
        axis_kde.text(threshold_value + 0.15, 70000, 'Hotspot Threshold: \n{}'.format(round(threshold_value, 4)),
                      color='red', size=15)

    # Set the yticks
    ytick_vals = np.arange(0, max(y_vals) + 20000, 20000).astype(np.int)
    axis_kde.set_yticks(ytick_vals)
    axis_kde.set_yticklabels(ytick_vals, size=18)

    # Set the xticks
    axis_kde.xaxis.set_tick_params(labelsize=18)

    # Set the top and right axis to be invisible:
    # https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
    axis_kde.spines['right'].set_visible(False)
    axis_kde.spines['top'].set_visible(False)

    # Set the axis label and save the figure to local directory
    axis_kde.set_xlabel('Kernel Density Values', size=20)
    axis_kde.set_ylabel('# of Spatial Units', size=20)
    figure.savefig(os.path.join(save_path, save_filename), bbox_inches='tight')


def kde_plot(density_data: pd.DataFrame, consider_sentiment: bool, draw_natural_break_bounds: bool,
             save_filename: str) -> None:
    """
    The histogram of density values outputed by ArcGIS
    :param density_data: the pandas dataframe used to save the density values
    :param consider_sentiment: consider sentiment or not in the kernel density computation or not
    :param draw_natural_break_bounds: draw the natural break verticle lines or not
    :param save_filename: the name of the saved figure
    :return: None. The histogram plot is saved to local directory
    """
    assert 'acc' in save_filename or 'cgs' in save_filename, 'The saved figure name should contain traffic type: ' \
                                                             'acc for accident and cgs for congestion'
    assert 'VALUE' in density_data, "The dataframe should contain columns saving the density valuess."
    assert 'COUNT' in density_data, "The dataframe should contain columns saving the number of cells " \
                                    "equalling a density value"

    # Create a list of kernel density values
    density_list = []
    for _, row in density_data.iterrows():
        density_list += [row['VALUE']] * np.int(row['COUNT'])
    final_dense_vals = [dense_val / 100 for dense_val in density_list]
    figure, axis_kde = plt.subplots(1, 1, figsize=(12, 8), dpi=300)

    # Draw the histogram
    if 'acc' in save_filename:  # If we are studying accident
        print('Creating the density histogram for Weibo accidents...')
        if consider_sentiment:
            axis_kde.set_title('Histogram of KDE Values to Detect Accident Hotspots\n(With Sentiment)', size=23)
        else:
            axis_kde.set_title('Histogram of KDE Values to Detect Accident Hotspots\n(Without Sentiment)', size=23)
        y_vals, x_vals, _ = axis_kde.hist(final_dense_vals, bins=50)
    else:  # If we are studying congestion
        print('Creating the density histogram for Weibo congestions...')
        if consider_sentiment:
            axis_kde.set_title('Histogram of KDE Values to Detect Congestion Hotspots\n(With Sentiment)', size=23)
        else:
            axis_kde.set_title('Histogram of KDE Values to Detect Congestion Hotspots\n(Without Sentiment)', size=23)
        y_vals, x_vals, _ = axis_kde.hist(final_dense_vals, bins=50)

    # Compute the threshold value
    threshold_value = round(np.mean(final_dense_vals) + 3 * np.std(final_dense_vals), 2)
    print('The threshold value is set to: {}'.format(threshold_value))

    # Plot the 5 natural breaks
    if draw_natural_break_bounds:
        # Compute the natural break vals list
        natural_breaks_val_list = jenkspy.jenks_breaks(final_dense_vals, nb_class=5)
        for index, bound in enumerate(natural_breaks_val_list):
            axis_kde.axvline(bound, linestyle='--', color='black')
        # Set the xticks for the natural breaks
        axis_kde.set_xticks(natural_breaks_val_list)
        natural_break_labels = [round(val, 2) for val in natural_breaks_val_list]
        axis_kde.set_xticklabels(natural_break_labels, size=18, rotation=45)

    # Set the yticks
    ytick_vals = np.arange(0, max(y_vals) + 10000, 10000).astype(np.int)
    axis_kde.set_yticks(ytick_vals)
    axis_kde.set_yticklabels(ytick_vals, size=18)

    # Draw the threshold line to determine the hotspot
    axis_kde.axvline(threshold_value, linestyle='--', color='red')
    axis_kde.text(threshold_value + 0.15, 80000, 'Hotspot Threshold: \n{}'.format(threshold_value), color='red',
                  size=15)

    # Set the axis label and save the figure to local directory
    axis_kde.set_xlabel('Kernel Density Values', size=20)
    axis_kde.set_ylabel('# of Spatial Units', size=20)
    figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def generate_wordcloud_density_classes_days(density_day_dict: dict, density_dataframe_dict: dict, traffic_type: str):
    """
    Generate wordcloud based on the time dictionary
    :param density_day_dict: a time dictionary saving the dates with big number of traffic accidents or congestions
    :param density_dataframe_dict: a dataframe dictionary saving the Weibo dataframes for given density class
    :param traffic_type: considered traffic event type. 'acc' for accident and 'cgs' for congestion
    :return: None. The wordcloud figures are saved to the local directory
    """
    assert traffic_type in ['acc', 'cgs'], 'The traffic info type should be given correctly'
    for key in density_day_dict:
        print('Coping with the {}'.format(key))
        studied_time_list = density_day_dict[key]
        studied_weibo_data = density_dataframe_dict[key]
        for day_tuple in studied_time_list:
            generate_wordcloud_in_given_day(dataframe=studied_weibo_data, month=day_tuple[0], day=day_tuple[1],
                                            save_filename='{}_wordcloud_{}_{}_{}.png'.format(
                                                traffic_type, key, day_tuple[0], day_tuple[1]))


def correlation_plot(dataframe, considered_column_list: list, save_filename: str) -> None:
    """
    Create the correlation plot for some columns of values in a dataframe
    All the values in the considered columns should be either int or float
    :param dataframe: the dataframe saving some values of some attributes.
    :param considered_column_list: a list containing the selected colnames
    :param save_filename: the name of the figure saving to local
    :return:
    """
    df = dataframe[considered_column_list]
    renamed_dict = {'acc_count': 'Accident Num', 'conges_count': 'Congestion Num', 'condition_count': 'Condition Num',
                    'sent_index': 'Sentiment Index', 'count': 'Count'}
    df_renamed = df.rename(columns=renamed_dict)
    palette = sns.color_palette("light:b", as_cmap=True)
    for colname in df_renamed:
        assert df_renamed[colname].dtype in ['float64', 'int64'], 'The data type of column {} is not right!'.format(
            colname)
    figure, axis = plt.subplots(1, 1, figsize=(25, 18), dpi=150)
    sns.heatmap(df_renamed.corr(), ax=axis, cmap=palette, annot=True)
    axis.set(yticks=np.arange(df_renamed.shape[1]) + 0.5)
    figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def sentiment_against_density_plot(hotspot_sent_density_data: pd.DataFrame,
                                   save_filename: str, dot_annotate=False):
    """
    Create the sentiment against density plot for each hotspot
    :param hotspot_sent_density_data: a pandas dataframe saving the sentiment index and density of each hotspot
    The dataframe is created by location_analysis.create_sentiment_against_density_data
    :param save_filename: the filename of the saved figure
    :param dot_annotate: whether use the hotspot id to annotate the dot or not
    :return: None. The scatter plot is saved to local directory
    """
    # Create the scatter plot
    assert 'acc' in save_filename or 'cgs' in save_filename, "The filename should contain either acc or cgs!"
    # Create dictionary for hotspot id and gmm clustering label, hotspot id and its color
    hotspot_sent_density_data['log_density'] = hotspot_sent_density_data.apply(
        lambda row: np.log(row['density']), axis=1)

    # Create the figure
    figure, axis = plt.subplots(1, 1, figsize=(15, 10), dpi=300)

    max_log_density, min_log_density = max(hotspot_sent_density_data['log_density']), \
                                       min(hotspot_sent_density_data['log_density'])
    max_ceil, min_floor = np.ceil(max_log_density), np.floor(min_log_density)
    if 'acc' in save_filename:
        axis.scatter(hotspot_sent_density_data['log_density'], hotspot_sent_density_data['sentiment'],
                     c=hotspot_sent_density_data['hotspot_id'], cmap='Paired')
    else:
        axis.scatter(hotspot_sent_density_data['log_density'], hotspot_sent_density_data['sentiment'],
                     c=hotspot_sent_density_data['hotspot_id'], cmap='Paired')

    # Annotate the dots in the scatter plot
    hotspot_ids = list(hotspot_sent_density_data['hotspot_id'])
    if dot_annotate:  # Whether annotate the dot or not
        text_list = []
        for index, hotspot_id in enumerate(hotspot_ids):
            x_loc = round(hotspot_sent_density_data['log_density'][index], 2)
            y_loc = round(hotspot_sent_density_data['sentiment'][index], 2)
            if index == 5:
                print('One pair of location: ({}, {})'.format(x_loc, y_loc))
            text_list.append(axis.annotate(str(hotspot_id), xy=(x_loc, y_loc), size=15))
        adjust_text(text_list, only_move={'points': 'y', 'texts': 'y'},
                    arrowprops=dict(arrowstyle="->", color='r', lw=1))

    # Plot the y = 0 horizontal line
    axis.axhline(y=0, linestyle='--', alpha=0.5, color='black')

    # Set xlabel and ylabel
    axis.set_xlabel('log(# of Weibos and Their Reposts)', size=20)
    axis.set_ylabel('Sentiment Index\n(% of Positive - % of Negative)', size=20)

    # Set the xticks and yticks
    axis.set_xticks(np.arange(min_floor, max_ceil + 0.5, 0.5))
    axis.xaxis.set_tick_params(labelsize=20)
    ytick_vals = np.arange(-1, 1.25, 0.25)
    ytick_labels = (ytick_vals * 100).astype(np.int)
    axis.set_yticks(ytick_vals)
    axis.set_yticklabels(ytick_labels, size=20)

    # # Set the legend
    # axis.legend(fontsize=20)

    # Delete unnecessary spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    # Save the figure
    figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def sentiment_against_density_plot_with_gmm(hotspot_sent_density_data: pd.DataFrame, hotspot_data_with_gmm_labels,
                                            save_filename: str, dot_annotate=False):
    """
    Create the sentiment against density plot for each hotspot
    :param hotspot_sent_density_data: a pandas dataframe saving the sentiment index and density of each hotspot
    The dataframe is created by location_analysis.create_sentiment_against_density_data
    :param hotspot_data_with_gmm_labels: a pandas dataframe saving the gmm labels for each hotspot
    :param save_filename: the filename of the saved figure
    :param dot_annotate: whether use the hotspot id to annotate the dot or not
    :return: None. The scatter plot is saved to local directory
    """
    # Create the scatter plot
    assert 'acc' in save_filename or 'cgs' in save_filename, "The filename should contain either acc or cgs!"
    assert 'gmm_label' in hotspot_data_with_gmm_labels, "The dataframe should have gmm cluster labels"

    # Create dictionary for hotspot id and gmm clustering label, hotspot id and its color
    hotspot_gmm_label_dict = {id: gmm_label for id, gmm_label in zip(hotspot_data_with_gmm_labels['hotspot_id'],
                                                                     hotspot_data_with_gmm_labels['gmm_label'])}
    hotspot_sent_density_data['log_density'] = hotspot_sent_density_data.apply(
        lambda row: np.log(row['density']), axis=1)
    hotspot_sent_density_data['gmm_label'] = hotspot_sent_density_data.apply(
        lambda row: hotspot_gmm_label_dict.get(row['hotspot_id'], -1), axis=1)
    colors = ['red', 'blue', 'green', 'black']
    color_dict = {id: color for id, color in zip([0, 1, 2, -1], colors)}
    hotspot_sent_density_data['color'] = hotspot_sent_density_data.apply(
        lambda row: color_dict[row['gmm_label']], axis=1)

    # Create the figure
    figure, axis = plt.subplots(1, 1, figsize=(15, 10), dpi=300)

    max_log_density, min_log_density = max(hotspot_sent_density_data['log_density']), \
                                       min(hotspot_sent_density_data['log_density'])
    max_ceil, min_floor = np.ceil(max_log_density), np.floor(min_log_density)
    if 'acc' in save_filename:
        axis.scatter(hotspot_sent_density_data['log_density'], hotspot_sent_density_data['sentiment'],
                     color=hotspot_sent_density_data['color'])
    else:
        axis.scatter(hotspot_sent_density_data['log_density'], hotspot_sent_density_data['sentiment'],
                     color=hotspot_sent_density_data['color'])

    # Annotate the dots in the scatter plot
    hotspot_ids = list(hotspot_sent_density_data['hotspot_id'])
    if dot_annotate:  # Whether annotate the dot or not
        text_list = []
        for index, hotspot_id in enumerate(hotspot_ids):
            x_loc = round(hotspot_sent_density_data['log_density'][index], 2)
            y_loc = round(hotspot_sent_density_data['sentiment'][index], 2)
            if index == 5:
                print('One pair of location: ({}, {})'.format(x_loc, y_loc))
            text_list.append(axis.annotate(str(hotspot_id), xy=(x_loc, y_loc), size=15))
        adjust_text(text_list, only_move={'points': 'y', 'texts': 'y'},
                    arrowprops=dict(arrowstyle="->", color='r', lw=1))

    # Plot the y = 0 horizontal line
    axis.axhline(y=0, linestyle='--', alpha=0.5, color='black')

    # Set xlabel and ylabel
    axis.set_xlabel('log(# of Weibos)', size=20)
    axis.set_ylabel('Sentiment Index\n(% of Positive - % of Negative)', size=20)

    # Set the xticks and yticks
    axis.set_xticks(np.arange(min_floor, max_ceil + 0.5, 0.5))
    axis.xaxis.set_tick_params(labelsize=20)
    ytick_vals = np.arange(-1, 1.25, 0.25)
    ytick_labels = (ytick_vals * 100).astype(np.int)
    axis.set_yticks(ytick_vals)
    axis.set_yticklabels(ytick_labels, size=20)

    # # Set the legend
    # axis.legend(fontsize=20)

    # Delete unnecessary spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    # Save the figure
    figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def plot_sent_dense(sent_dense_data: pd.DataFrame, traffic_type: str, save_filename: str) -> None:
    """
    Plot the sentiment against number of Weibos for each hotspot
    :param sent_dense_data: a pandas dataframe saving the sentiment index and number of Weibos in hotspots
    :param traffic_type: the studied traffic type
    :param save_filename: the saved filename
    :return: None. The created figure is saved to local directory
    """
    assert traffic_type in {'acc', 'cgs'}, "Please specify a correct traffic type."

    hotspot_ids = list(sent_dense_data['hotspot_id'])
    count_list = list(sent_dense_data['density'])
    count_log_list = [np.log(count) for count in count_list]
    sentiment_list = list(sent_dense_data['sentiment'])

    # Get the 50%, 75% percentile of sentiment and log density
    count_50, count_75 = np.percentile(count_log_list, 50), np.percentile(count_log_list, 75)
    sent_50, sent_25 = np.percentile(sentiment_list, 50), np.percentile(sentiment_list, 25)

    # Draw the hotspots with different colors
    figure, axis = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
    priority_one_set, priority_two_set = set(), set()
    for count_log, sent, hotspot_id in zip(count_log_list, sentiment_list, hotspot_ids):
        if (count_log >= count_75) and (sent <= sent_25):
            axis.scatter(count_log, sent, color='red', s=20)
            priority_one_set.add(hotspot_id)
        elif (count_50 <= count_log) and (sent <= sent_50) and (hotspot_id not in priority_one_set):
            axis.scatter(count_log, sent, color='red', s=20)
            priority_two_set.add(hotspot_id)
        else:
            axis.scatter(count_log, sent, color='grey', s=20)

    # Annotate the dots
    for i, txt in enumerate(hotspot_ids):
        axis.annotate(txt, (count_log_list[i], sentiment_list[i]), size=12)

    # axis.axhline(sent_50, color='blue', linestyle='--', alpha=0.5)
    # axis.axvline(count_50, color='blue', linestyle='--', alpha=0.5)
    # axis.axhline(sent_25, color='red', linestyle='--', alpha=0.5)
    # axis.axvline(count_75, color='red', linestyle='--', alpha=0.5)

    # Add the x label and y label
    if traffic_type == 'acc':
        axis.set_xlabel('log(# of Accident-relevant Weibos)', size=25)
    else:
        axis.set_xlabel('log(# of Congestion-relevant Weibos)', size=25)
    axis.set_ylabel('Sentiment Index\n(% of Positive - % of Negative)', size=25)

    # Plot the rectangles
    ytick_vals = axis.get_yticks().tolist()
    rec_priority = Rectangle(xy=(count_50, ytick_vals[0]),
                                 width=(max(count_log_list) - count_50), height=(sent_50 - ytick_vals[0]),
                                 facecolor='red', alpha=0.15)
    # rec_priority_two_first = Rectangle(xy=(count_50, min_y),
    #                                    width=(count_75 - count_50), height=(sent_50 - min_y),
    #                                    facecolor='blue', alpha=0.3)
    # rec_priority_two_second = Rectangle(xy=(count_75, sent_25),
    #                                     width=(max(count_log_list) - count_75), height=(sent_50 - sent_25),
    #                                     facecolor='blue', alpha=0.3)
    axis.add_patch(rec_priority)
    # axis.add_patch(rec_priority_two_first)
    # axis.add_patch(rec_priority_two_second)

    # Edit the spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    # Edit the xticks and yticks
    ytick_vals_round_string = [str(round(tick_val, 2) * 100) for tick_val in ytick_vals]
    axis.set_yticks(ytick_vals)
    axis.set_yticklabels(ytick_vals_round_string)
    axis.xaxis.set_tick_params(labelsize=20)
    axis.yaxis.set_tick_params(labelsize=20)

    # Output the considered hotspots with different priority
    print('Priority one set: {}'.format(priority_one_set))
    print('Priority two set: {}'.format(priority_two_set))

    figure.savefig(os.path.join(figures_path, save_filename), bbox_inches='tight')


def plot_validation_daily_alarm_detection(daily_compare_dataframe: pd.DataFrame, save_path: str, save_filename: str):
    """
    Conduct the hotspot validation on a daily basis, using false alarm rate and miss detection rate
    :param daily_compare_dataframe: a pandas dataframe saving the metrics for hotspot validation daily
    :param save_path: the path used to save the created dataframe
    :param save_filename: the saved file name
    :return: None. The figure is saved to local directory
    """
    assert 'acc' in save_filename or 'cgs' in save_filename, "The saved filename should contain traffic type info"

    # Assign the value to traffic type
    if 'acc' in save_filename:
        traffic_event_type = 'acc'
    else:
        traffic_event_type = 'cgs'

    # Prepare the dataframe to validation the hotspot identification module in specific days
    daily_compare_dataframe['datetime_obj'] = daily_compare_dataframe.apply(
        lambda row: datetime(2012, row['month'], row['day']), axis=1)
    daily_compare_dataframe_sorted = daily_compare_dataframe.sort_values(by='datetime_obj')
    daily_compare_dataframe_sorted_select = daily_compare_dataframe_sorted.loc[
        daily_compare_dataframe_sorted['miss_detection_actual'] != 'no actual'].reset_index(
        drop=True).copy()
    daily_compare_dataframe_sorted_select['miss_detection_actual'] = daily_compare_dataframe_sorted_select[
        'miss_detection_actual'].astype(np.float64)
    month_list, day_list = list(daily_compare_dataframe_sorted_select['month']), list(
        daily_compare_dataframe_sorted_select['day'])
    date_list = ['{}-{}'.format(month, day) for month, day in zip(month_list, day_list)]

    # Plot the false alarm, miss detection, and PAI indices
    figure, axes = plt.subplots(2, 2, figsize=(25, 16), dpi=300)
    axes[0][0].plot(daily_compare_dataframe_sorted_select['false_alarm_weibo'], label='False Alarm (Weibo)',
                    color='blue')
    # x_text = 3
    # y_text = np.percentile(daily_compare_dataframe_sorted_select['false_alarm_weibo'], 60)
    # text_string = 'Mean: {}'.format(round(np.mean(daily_compare_dataframe_sorted_select['false_alarm_weibo']), 4))
    # axes[0][0].text(x_text, y_text, text_string)
    axes[0][0].axhline(np.mean(daily_compare_dataframe_sorted_select['false_alarm_weibo']), alpha=0.5, color='black',
                       linestyle='--')
    axes[0][0].set_ylabel('False Alarm (Weibo) %', size=20)

    axes[1][0].plot(daily_compare_dataframe_sorted_select['false_alarm_actual'], label='False Alarm (Actual)',
                    color='blue')
    # x_text = 3
    # y_text = np.percentile(daily_compare_dataframe_sorted_select['false_alarm_actual'], 60)
    # text_string = 'Mean: {}'.format(round(np.mean(daily_compare_dataframe_sorted_select['false_alarm_actual']), 4))
    # axes[1][0].text(x_text, y_text, text_string)
    axes[1][0].axhline(np.mean(daily_compare_dataframe_sorted_select['false_alarm_actual']), alpha=0.5, color='black',
                       linestyle='--')
    axes[1][0].set_ylabel('False Alarm (Actual) %', size=20)

    axes[0][1].plot(daily_compare_dataframe_sorted_select['miss_detection_weibo'], label='Miss Detection (Weibo)',
                    color='red')
    # x_text = 3
    # y_text = np.percentile(daily_compare_dataframe_sorted_select['miss_detection_weibo'], 60)
    # text_string = 'Mean: {}'.format(round(np.mean(daily_compare_dataframe_sorted_select['miss_detection_weibo']), 4))
    # axes[0][1].text(x_text, y_text, text_string)
    axes[0][1].axhline(np.mean(daily_compare_dataframe_sorted_select['miss_detection_weibo']), alpha=0.5, color='black',
                       linestyle='--')
    axes[0][1].set_ylim(0, 1)
    axes[0][1].set_ylabel('Miss Detection (Weibo) %', size=20)

    axes[1][1].plot(daily_compare_dataframe_sorted_select['miss_detection_actual'], label='Miss Detection (Actual)',
                    color='red')
    # x_text = 3
    # y_text = np.percentile(daily_compare_dataframe_sorted_select['miss_detection_actual'], 60)
    # text_string = 'Mean: {}'.format(round(np.mean(daily_compare_dataframe_sorted_select['miss_detection_actual']), 4))
    # axes[1][1].text(x_text, y_text, text_string)
    axes[1][1].axhline(np.mean(daily_compare_dataframe_sorted_select['miss_detection_actual']), alpha=0.5,
                       color='black',
                       linestyle='--')
    axes[1][1].set_ylim(0, 1)
    axes[1][1].set_ylabel('Miss Detection (Actual) %', size=20)

    # Annotate the xticks and yticks of each axis
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if traffic_event_type is 'acc':
                axes[i][j].set_xticks(list(range(daily_compare_dataframe_sorted_select.shape[0])))
                axes[i][j].set_xticklabels(date_list, rotation=45, size=15)
            else:
                xtick_vals = np.array(list(range(daily_compare_dataframe_sorted_select.shape[0])))
                # indices = np.array(list(range(len(xtick_vals))))
                new_indices = np.linspace(start=0, stop=len(xtick_vals) - 1, num=10, dtype=int)
                new_xtick_vals = xtick_vals[new_indices]
                new_xtick_labels = [date_list[i] for i in new_indices]
                axes[i][j].set_xticks(new_xtick_vals)
                axes[i][j].set_xticklabels(new_xtick_labels, rotation=45, size=15)
            axes[i][j].yaxis.set_tick_params(labelsize=18)
            axes[i][j].legend(fontsize=25)
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['top'].set_visible(False)
            # Plot the yticks
            ytick_vals = axes[i][j].get_yticks().tolist()
            ytick_vals_for_plot = [val for val in ytick_vals if val >= 0]
            ytick_vals_round_string = [str(round(tick_val * 100, 4)) for tick_val in ytick_vals_for_plot]
            axes[i][j].set_yticks(ytick_vals_for_plot)
            axes[i][j].set_yticklabels(ytick_vals_round_string)

    figure.savefig(os.path.join(save_path, save_filename), bbox_inches='tight')


def plot_validation_daily_pai(daily_compare_dataframe: pd.DataFrame, save_path: str, save_filename: str,
                              plot_pai_ratio: bool = True):
    """
    Conduct the hotspot validation on a daily basis using PAI values (PAI weibo, PAI actual, PAI ratio)
    :param daily_compare_dataframe: a pandas dataframe saving the metrics for hotspot validation daily
    :param save_path: the path used to save the created dataframe
    :param save_filename: the saved file name
    :param plot_pai_ratio: boolean. If True, plot the PAI ratio; else, plot hotspot precision
    :return: None. The figure is saved to local directory
    """
    assert 'acc' in save_filename or 'cgs' in save_filename, "The saved filename should contain traffic type info"

    # Assign the value to traffic type
    if 'acc' in save_filename:
        traffic_type = 'acc'
    else:
        traffic_type = 'cgs'

    # Prepare the dataframe to validation the hotspot identification module in specific days
    daily_compare_dataframe['datetime_obj'] = daily_compare_dataframe.apply(
        lambda row: datetime(2012, row['month'], row['day']), axis=1)
    daily_compare_dataframe_sorted = daily_compare_dataframe.sort_values(by='datetime_obj')
    daily_compare_dataframe_sorted_select = daily_compare_dataframe_sorted.loc[
        daily_compare_dataframe_sorted['miss_detection_actual'] != 'no actual'].reset_index(
        drop=True).copy()
    daily_compare_dataframe_sorted_select['miss_detection_actual'] = daily_compare_dataframe_sorted_select[
        'miss_detection_actual'].astype(np.float64)
    month_list, day_list = list(daily_compare_dataframe_sorted_select['month']), list(
        daily_compare_dataframe_sorted_select['day'])
    date_list = ['{}-{}'.format(month, day) for month, day in zip(month_list, day_list)]

    # Plot the false alarm, miss detection, and PAI indices
    figure, axes = plt.subplots(3, 1, figsize=(20, 16), dpi=300, sharex=True)

    axes[0].plot(daily_compare_dataframe_sorted_select['PAI_actual'], label='PAI (Actual)', color='green')
    # x_text = 3
    # y_text = np.percentile(daily_compare_dataframe_sorted_select['PAI_actual'], 60)
    # text_string = 'Mean: {}'.format(round(np.mean(daily_compare_dataframe_sorted_select['PAI_actual']), 4))
    # axes[1][2].text(x_text, y_text, text_string)
    axes[0].axhline(np.mean(daily_compare_dataframe_sorted_select['PAI_actual']), alpha=0.5, color='black',
                       linestyle='--')
    axes[0].set_ylabel('PAI(Actual)', size=20)

    axes[1].plot(daily_compare_dataframe_sorted_select['PAI_weibo'], label='PAI (Weibo)', color='green')
    # x_text = 3
    # y_text = np.percentile(daily_compare_dataframe_sorted_select['PAI_weibo'], 60)
    # text_string = 'Mean: {}'.format(round(np.mean(daily_compare_dataframe_sorted_select['PAI_weibo']), 4))
    # axes[0][2].text(x_text, y_text, text_string)
    axes[1].axhline(np.mean(daily_compare_dataframe_sorted_select['PAI_weibo']), alpha=0.5, color='black',
                    linestyle='--')
    axes[1].set_ylabel('PAI(Weibo)', size=20)

    if plot_pai_ratio:
        axes[2].plot(daily_compare_dataframe_sorted_select['PAI_ratio'], color='red',
                        linestyle='--', label=r'PAI Ratio')
        axes[2].axhline(np.mean(daily_compare_dataframe_sorted_select['PAI_ratio']), alpha=0.5, color='black',
                           linestyle='--')
        axes[2].set_ylabel('PAI Ratio', size=20)
    else:
        axes[2].set_ylabel('Hotspot Precision %', size=20)
        axes[2].plot(daily_compare_dataframe_sorted_select['hotspot_precision'], color='red',
                     linestyle='--', label=r'Hotspot Precision')
        axes[2].axhline(np.mean(daily_compare_dataframe_sorted_select['hotspot_precision']), alpha=0.5, color='black',
                        linestyle='--')
        ytick_vals = axes[2].get_yticks().tolist()
        ytick_vals_for_plot = [val for val in ytick_vals if val >= 0]
        ytick_vals_round_string = [str(round(tick_val * 100, 0)) for tick_val in ytick_vals_for_plot]
        print(ytick_vals_round_string)
        axes[2].set_yticks(ytick_vals_for_plot)
        axes[2].set_yticklabels(ytick_vals_round_string)

    for i in range(axes.shape[0]):
        if traffic_type is 'acc':
            axes[i].set_xticks(list(range(daily_compare_dataframe_sorted_select.shape[0])))
            axes[i].set_xticklabels(date_list, rotation=45, size=15)
        else:
            xtick_vals = np.array(list(range(daily_compare_dataframe_sorted_select.shape[0])))
            # indices = np.array(list(range(len(xtick_vals))))
            new_indices = np.linspace(start=0, stop=len(xtick_vals) - 1, num=10, dtype=int)
            new_xtick_vals = xtick_vals[new_indices]
            new_xtick_labels = [date_list[i] for i in new_indices]
            axes[i].set_xticks(new_xtick_vals)
            axes[i].set_xticklabels(new_xtick_labels, rotation=45, size=15)
        axes[i].yaxis.set_tick_params(labelsize=18)
        axes[i].legend(fontsize=25)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)

    figure.savefig(os.path.join(save_path, save_filename), bbox_inches='tight')


if __name__ == '__main__':

    # Draw the kde density values and thresholds for hotspot identification
    print('Draw the kernel density values and thresholds for hotspot identification...')
    fishnet_shapefiles = [file for file in os.listdir(data_paths.raster_fishnet) if file.endswith('.shp')]
    print(fishnet_shapefiles)
    for fishnet_file in fishnet_shapefiles:
        bandwidth = fishnet_file[:-4].split('_')[1]
        if int(bandwidth) == 2000:
            print('Coping with the fishnet file: {}'.format(fishnet_file))
            fishnet_shape = gpd.read_file(os.path.join(data_paths.raster_fishnet, fishnet_file), encoding='utf-8')
            density_val = list(fishnet_shape['raster_val'])
            if 'without_sent' in fishnet_file:
                consider_sent = False
            else:
                consider_sent = True
            print(consider_sent)
            traffic_type = fishnet_file[:-4].split('_')[0]
            unit_size= fishnet_file[:-4].split('_')[2]
            threshold_value = np.mean(density_val) + 3 * np.std(density_val)
            print("traffic type: {}; bandwidth: {}; unit size: {}; threshold value: {}".format(
                traffic_type, bandwidth, unit_size, threshold_value))
            draw_kde_densities(density_values=density_val, consider_sentiment=consider_sent,
                               threshold_value=threshold_value, save_filename=fishnet_file[:-4]+'.png',
                               spatial_unit_size=int(unit_size))

    # Generate plots for hotspot comparison
    acc_hotspot_compare_data = pd.read_csv(os.path.join(data_paths.kde_analysis, 'kde_compare', 'acc_kde_compare.csv'),
                                           index_col=0, encoding='utf-8')
    cgs_hotspot_compare_data = pd.read_csv(os.path.join(data_paths.kde_analysis, 'kde_compare', 'cgs_kde_compare.csv'),
                                           index_col=0, encoding='utf-8')
    acc_hotspot_compare_data_select = acc_hotspot_compare_data.loc[acc_hotspot_compare_data['bandwidth'] == 2000]
    cgs_hotspot_compare_data_select = cgs_hotspot_compare_data.loc[cgs_hotspot_compare_data['bandwidth'] == 2000]
    acc_dict_actual, acc_dict_weibo = compare_hotspot_roc(hotspot_compare_dataframe=acc_hotspot_compare_data_select,
                                                          considered_traffic_type='acc')
    cgs_dict_actual, cgs_dict_weibo = compare_hotspot_roc(hotspot_compare_dataframe=cgs_hotspot_compare_data_select,
                                                          considered_traffic_type='cgs')
    print(acc_dict_actual)
    print('*'*20)
    print(acc_dict_weibo)
    print('\n'+'=' * 20)
    print(cgs_dict_actual)
    print('*' * 20)
    print(cgs_dict_weibo)

    # Validate the hotspot on a daily basis
    acc_validate_daily = pd.read_csv(os.path.join(data_paths.kde_analysis, 'kde_compare',
                                                  'acc_daily_validate_three_std.csv'),
                                     index_col=0,
                                     encoding='utf-8')
    cgs_validate_daily = pd.read_csv(os.path.join(data_paths.kde_analysis, 'kde_compare',
                                                  'cgs_daily_validate_three_std.csv'),
                                     index_col=0,
                                     encoding='utf-8')
    plot_validation_daily_alarm_detection(daily_compare_dataframe=acc_validate_daily,
                                          save_path=data_paths.figures_path, save_filename='acc_daily_validate.png')
    plot_validation_daily_alarm_detection(daily_compare_dataframe=cgs_validate_daily,
                                          save_path=data_paths.figures_path, save_filename='cgs_daily_validate.png')
    plot_validation_daily_pai(daily_compare_dataframe=acc_validate_daily,
                              save_path=data_paths.figures_path, save_filename='acc_daily_pai.png',
                              plot_pai_ratio=False)
    plot_validation_daily_pai(daily_compare_dataframe=cgs_validate_daily,
                              save_path=data_paths.figures_path, save_filename='cgs_daily_pai.png',
                              plot_pai_ratio=False)
