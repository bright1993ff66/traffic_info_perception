# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 18:54
# @Author  : Haoliang Chang
import itertools
import os
import re
import csv
from collections import Counter
from datetime import datetime
from random import sample
import json
from json.decoder import JSONDecodeError

import networkx as nx
import numpy as np
import pandas as pd
import pytz
from colour import Color
from geopy.distance import geodesic
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
import geopandas as gpd

from data_paths import kde_analysis, shapefile_path

kde_compare_path = os.path.join(kde_analysis, 'kde_compare')

# Specify the timezone in Shanghai
timezone_shanghai = pytz.timezone('Asia/Shanghai')

# traffic-related word dictionary
traffic_word_set_update = {'堵', '拥堵', '阻塞', '塞车', '拥挤', '车祸', '剐蹭', '事故', '撞', '追尾', '相撞', '路况', '路段',
                           '路线', '封道', '封路', '绕行', '畅通', '立交', '高架', '快速路', '大桥', '隧道', '驾驶', '避让',
                           '车距'}
congestion_traffic_word_set = {'堵', '拥堵', '阻塞', '塞车', '拥挤'}
accident_traffic_word_set = {'车祸', '剐蹭', '事故', '撞', '追尾', '相撞'}
# the user id of official traffic informaiton social media accounts in Shanghai
# 1980308627: 乐行上海:
# https://weibo.com/lexingsh?from=page_100206_profile&wvr=6&mod=bothfollow&refer_flag=1005050010_&is_all=1
# 1750349294: 上海交通广播
# https://weibo.com/shjtgb?is_all=1
# 1976304153: 上海市交通委员会
# https://weibo.com/p/1001061976304153/home?from=page_100106&mod=TAB&is_hot=1#place
traffic_acount_uid = {1980308627, 1750349294, 1976304153}
# some selected accounts for analysis
# 2539961154: 上海发布
# https://weibo.com/shanghaicity?topnav=1&wvr=6&topsug=1&is_hot=1
# 1763251165: 宣克灵
# https://weibo.com/xuankejiong?topnav=1&wvr=6&topsug=1&is_hot=1
# 2256231983: 上海热门资讯
# https://weibo.com/209273419?topnav=1&wvr=6&topsug=1&is_hot=1
# 1971537652: 魔都生活圈
# https://weibo.com/u/1971537652?topnav=1&wvr=6&topsug=1&is_hot=1
traffic_acount_uid_final = {2539961154, 1763251165, 2256231983, 1971537652}

# Specify the random seed
random_seed = 7
np.random.seed(random_seed)

# Set the standard scaler
scaler = StandardScaler()

# Load the shanghai shape
shanghai_shape = gpd.read_file(os.path.join(shapefile_path, 'overall', 'shanghai_proj_utm_51.shp'),
                               encoding='utf-8')


def delete_user(text):
    """
    Delete the @user in the weibo or tweet text
    :param text: a weibo or tweet string
    :return: a text string without @user
    """
    result_text = re.sub("@[^，，：：\s@:]+", "", text)
    return result_text


def get_url(text_string):
    """
    Get the url of a Weibo given a Weibo string
    :param text_string: a Weibo text string
    :return: the url of the studied Weibo string
    """
    if not isinstance(text_string, str):  # Ensure that the input is a string
        return "No URL"
    elif 'http' not in text_string:  # If the string does not contain url
        return "No URL"
    else:  # Return the url of the studied Weibo
        return 'http' + text_string.split("http", 1)[1]


def merge_dict(sum_dict, a_dict):
    """
    Merge a sum dictionary and a dictionary for a csv file
    Args:
        sum_dict: the sum_dict records the total number of tweets found in one city
        a_dict: a count dict for a csv tweet file
    Returns: a sum_dict which has added values from a_dict
    """
    if a_dict == Counter():
        return sum_dict
    for key in a_dict:
        sum_dict[key] += a_dict[key]
    return sum_dict


def rename_columns(studied_weibo_dataframe: pd.DataFrame or gpd.geodataframe.GeoDataFrame) -> pd.DataFrame:
    """
    Rename the columns of a dataframe to a structured format
    :param studied_weibo_dataframe: a studied weibo dataframe
    :return: the reformatted weibo dataframe
    """
    if 'retweete_1' in studied_weibo_dataframe:  # cope with the output from the silly arcmap
        rename_dict_info = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost',
                            'retweete_1': 'retweeters_text'}
        renamed_data = studied_weibo_dataframe.rename(columns=rename_dict_info)
    else:
        renamed_data = studied_weibo_dataframe.copy()

    if 'loc_lat' in renamed_data:  # For latitude and longitude information
        dropped_data = renamed_data.drop(columns=['lat', 'lon'])
        rename_dict_loc = {'loc_lon': 'lon', 'loc_lat': 'lat'}
        final_data = dropped_data.rename(columns=rename_dict_loc)
    else:
        final_data = renamed_data.copy()

    if 'sent_repost' in final_data or 'retweeters_ids' in final_data:  # match the previous output
        final_data_renamed = final_data.rename(columns={'sent_repost': 'sent_repos',
                                                        'retweeters_ids': 'retweeters',
                                                        'retweets_id': 'retweet_ids'})
    else:
        final_data_renamed = final_data.copy()

    if 'Name' in final_data:  # If the district information is provided
        considered_columns = ['index_val', 'author_id', 'weibo_id', 'created_at', 'text', 'lat',
                              'lon', 'retweeters', 'retweeters_text', 'retweet_ids', 'local_time',
                              'year', 'month', 'traffic_weibo', 'traffic_repost', 'Name', 'datatype',
                              'traffic_type']
    else:  # If the district information is not provided
        considered_columns = ['index_val', 'author_id', 'weibo_id', 'created_at', 'text', 'lat',
                              'lon', 'retweeters', 'retweeters_text', 'retweet_ids', 'local_time',
                              'year', 'month', 'traffic_weibo', 'traffic_repost', 'datatype',
                              'traffic_type']
    return final_data_renamed[considered_columns]


def assign_sentiment_info(studied_weibo_data: pd.DataFrame) -> pd.DataFrame:
    """
    Get the "sent_val" column in the KDE analysis
    :param studied_weibo_data: one traffic-related Weibo data
    :return: an updated dataframe with another "sent_val" column for the following KDE analysi
    """
    assert 'sent_weibo' in studied_weibo_data, "The dataframe should contain the sentiment of Weibo."
    studied_weibo_data_copy = studied_weibo_data.copy()

    def get_sent_val(sent_value: int) -> int:
        if sent_value == 0:
            return 1
        else:
            return 0

    studied_weibo_data_copy['sent_val'] = studied_weibo_data_copy.apply(lambda row: get_sent_val(row['sent_weibo']),
                                                                        axis=1)
    return studied_weibo_data_copy


def normalize_data(dataframe: pd.DataFrame, select_columns: str, density_threshold: float):
    """
    Normalize the selected columns of a pandas dataframe.
    Each selected column would be normalized with mean 0 and std 1
    :param dataframe: a pandas dataframe to be normalized
    :param select_columns: the name of the select columns
    :param density_threshold: Rows with density more than density_threshold are considered
    :return: a pandas dataframe with normalized data
    """
    assert 'density' in dataframe, 'The dataframe should have one column saving the density value!'
    data_select = dataframe.loc[dataframe['density'] > density_threshold]
    data_select_copy = data_select.copy()
    data_select_copy[select_columns] = scaler.fit_transform(data_select_copy[select_columns].to_numpy())
    data_final = data_select_copy.reset_index(drop=True)
    return data_final


def describe_traffic_type(dataframe: pd.DataFrame):
    """
    Give descriptive statistics of traffic type
    :param dataframe: a pandas dataframe saving the traffic-related messages (either Weibo or actual records)
    :return: None. The descriptive statistics are printed
    """
    assert 'traffic_type' in dataframe, "The dataframe should have a column named 'traffic_type'"
    traffic_type_counter = Counter(dataframe['traffic_type'])
    print('Sum: {}; Accident-related: {}; Congestion-related: {}'.format(
        sum(traffic_type_counter.values()), traffic_type_counter['accident'], traffic_type_counter['congestion']))


def transform_string_time_to_datetime(time_string, target_time_zone, convert_utc_time=False):
    """
    Transform the string time to the datetime
    :param time_string: a time string
    :param target_time_zone: the target time zone
    :param convert_utc_time: whether transfer the datetime object to utc first. This is true when the
    time string is recorded as the UTC time
    :return: a structured datetime object
    """
    datetime_object = datetime.strptime(time_string, '%a %b %d %H:%M:%S %z %Y')
    if convert_utc_time:
        final_time_object = datetime_object.replace(tzinfo=pytz.utc).astimezone(target_time_zone)
    else:
        final_time_object = datetime_object.astimezone(target_time_zone)
    return final_time_object


def transform_datetime_string_to_datetime(string, target_timezone, source_timezone=timezone_shanghai):
    """
    Transform a datetime string to the corresponding datetime. The timezone is in +8:00
    :param string: the string which records the time of the posted tweets(this string's timezone is HK time)
    :param target_timezone: the target timezone datetime object
    :param source_timezone: the source timezone of datetime string, default: pytz.timezone("Asia/Shanghai")
    :return: a datetime object which could get access to the year, month, day easily
    """
    datetime_object = datetime.strptime(string, '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=source_timezone)
    if source_timezone != target_timezone:
        final_time_object = datetime_object.replace(tzinfo=target_timezone)
    else:
        final_time_object = datetime_object
    return final_time_object


def encode_time(time_string, target_timezone=timezone_shanghai):
    """
    Encode the time string in the official traffic accident dataframe
    :param time_string: a string records the time
    :param target_timezone: the target timezone
    :return: the datetime object
    """
    date_str, time_str = time_string.split(' ')
    if '/' in time_string:
        year, month, day = date_str.split('/')
        hour, minute = time_str.split(':')
    elif '-' in time_string:
        year, month, day = date_str.split('-')
        hour, minute, _ = time_str.split(':')
    else:
        raise ValueError('Something wrong with the time string. Use datetime package instead!')
    return datetime(int(year), int(month), int(day), int(hour), int(minute), 0, tzinfo=target_timezone)


def get_tpr_fpr_for_used_threshold(traffic_type: str, for_weibo: bool) -> dict:
    assert traffic_type in {'acc', 'cgs'}, "The traffic type should be either 'acc' or 'cgs'"

    if traffic_type == 'acc':
        compare_dataframe = pd.read_csv(os.path.join(kde_compare_path, 'acc_kde_compare_given_threshold.csv'),
                                        index_col=0, encoding='utf-8')
    else:
        compare_dataframe = pd.read_csv(os.path.join(kde_compare_path, 'cgs_kde_compare_given_threshold.csv'),
                                        index_col=0, encoding='utf-8')
    if for_weibo:
        result_dict = {(row.consider_sent, str(row.unit_size)): (row.TPR_weibo, row.FPR_weibo)
                       for _, row in compare_dataframe.iterrows()}
    else:
        result_dict = {(row.consider_sent, str(row.unit_size)): (row.TPR_actual, row.FPR_actual)
                       for _, row in compare_dataframe.iterrows()}
    return result_dict


def spatial_join(weibo_gdf: gpd.geodataframe, shape_area: gpd.geodataframe) -> gpd.geodataframe:
    """
    Find the tweets posted in one city's open space
    :param weibo_gdf: the geopandas dataframe saving the tweets
    :param shape_area: the shapefile of a studied area, such as city, open space, etc
    :return: tweets posted in open space
    """
    assert weibo_gdf.crs == shape_area.crs, 'The coordinate systems do not match!'
    joined_data = gpd.sjoin(left_df=weibo_gdf, right_df=shape_area, op='within')
    joined_data_final = joined_data.drop_duplicates(subset=['weibo_id'])
    return joined_data_final


def assign_districts(weibo_dataframe: pd.DataFrame, shape_area: gpd.geodataframe):
    """
    Assign the districts based on the pandas Weibo dataframe and the area shapefile
    :param weibo_dataframe: the pandas dataframe saving the Weibo data
    :param shape_area: the geopandas shapefile of the studied area
    :return: the weibo dataframe with district information
    """
    assert 'Name' in shape_area, "The area shapefile should contain the name of districts"
    # assert 'Eng_Name' in shape_area, "The area shapefile should contain the English name of districts"

    used_crs_epsg = shape_area.crs.to_epsg()
    weibo_dataframe_reformat = rename_columns(studied_weibo_dataframe=weibo_dataframe)
    weibo_shapefile = gpd.GeoDataFrame(
        weibo_dataframe_reformat, geometry=gpd.points_from_xy(
            weibo_dataframe_reformat.lon, weibo_dataframe_reformat.lat)).set_crs(epsg=4326).to_crs(epsg=used_crs_epsg)
    shape_area_geometry = shape_area[['geometry', 'Name']]
    weibo_select = spatial_join(weibo_gdf=weibo_shapefile, shape_area=shape_area_geometry)
    return weibo_select


def get_dataframe_in_gmm_clusters(dataframe: pd.DataFrame, cluster_id_dict: dict, save_path: str,
                                  traffic_event_type: str):
    """
    Get the dataframe for each gmm cluster
    :param dataframe: a pandas dataframe saving the hotspot index information for each Weibo
    :param cluster_id_dict: a Python dict. The key is the cluster name and the value is the hotspot ids
    :param save_path: the saving path of the created dataframe
    :param traffic_event_type: the type of traffic event we consider
    :return: None. The dataframe is saved to the local directory
    """
    assert 'hotspot_id' in dataframe, "The dataframe should have one column for hotspots"
    assert 'acc' in traffic_event_type or 'cgs' in traffic_event_type, \
        "The traffic event type must contain either 'acc' or 'cgs'"
    print('Coping with {} relevant Weibos'.format(traffic_event_type))
    for cluster_name in cluster_id_dict:
        cluster_set = cluster_id_dict[cluster_name]
        cluster_dataframe = dataframe.loc[dataframe['hotspot_id'].isin(cluster_set)]
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(cluster_dataframe,
                                                                              repost_column='retweeters_text')
        total_count = pos_count + neutral_count + neg_count
        print('For {}, we have got {} Weibos and their reposts'.format(cluster_name, total_count))
        cluster_dataframe.to_csv(os.path.join(save_path, traffic_event_type + '_gmm_{}.csv'.format(cluster_name)))


def get_traffic_dataframes(dataframe: pd.DataFrame):
    """
    Get different type of Weibo dataframe:
    - geocoded Weibos with location information in the Weibo text
    - geocoded Weibos with location information in the repost text
    - nongeocoded Weibos with location information in the Weibo text
    - nongeocoded Weibos with location information in the repost text
    The code should consider both the Weibo and repost data (if available)
    :param dataframe: a Weibo dataframe
    :return: four pandas dataframes: geocoded weibo, geocoded repost, nongeocoded weibo, nongeocoded repost
    """
    assert 'datatype' in dataframe, 'Make sure we have a column saving the type of the dataframe'
    # Cope with the Weibos with self-reported geo-information
    dataframe_geocoded = dataframe.loc[dataframe['datatype'] == 'geocoded']
    geocoded_weibo = dataframe_geocoded.loc[(dataframe_geocoded['traffic_weibo'].isin([1, 2])) & (
        dataframe_geocoded['traffic_repost'].isin([0, 1]))]
    geocoded_repost = dataframe_geocoded.loc[dataframe_geocoded['traffic_repost'] == 2]
    # Cope with the Weibos without geo-information
    dataframe_not_geocoded = dataframe.loc[dataframe['datatype'] != 'geocoded']
    nongeocoded_weibo = dataframe_not_geocoded.loc[(dataframe_not_geocoded['traffic_weibo'] == 2) & (
        dataframe_not_geocoded['traffic_repost'].isin([0, 1]))]
    nongeocoded_repost = dataframe_not_geocoded.loc[dataframe_not_geocoded['traffic_repost'] == 2]
    return geocoded_weibo, geocoded_repost, nongeocoded_weibo, nongeocoded_repost


def create_weibo_count_table(traffic_weibo_dataframe: pd.DataFrame):
    """
    Create a descriptive statistics information about the traffic Weibo dataframe
    :param traffic_weibo_dataframe: a traffic Weibo dataframe
    :return: None. A descriptive statistics about the number of traffic relevant Weibos based on
    geocoded/nongeocoded, accident/congestion/other will be printed
    """
    assert 'datatype' in traffic_weibo_dataframe, 'The information about whether a Weibo is geocoded should be included'
    assert 'traffic_type' in traffic_weibo_dataframe, 'The type of the traffic information should be included'
    geocoded_weibo, geocoded_repost, nongeocoded_weibo, nongeocoded_repost = get_traffic_dataframes(
        traffic_weibo_dataframe)
    geocoded_dataframe_list = [geocoded_weibo, geocoded_repost]
    nongeocoded_dataframe_list = [nongeocoded_weibo, nongeocoded_repost]
    geocoded_acc_count, geocoded_cgs_count, geocoded_other_count = 0, 0, 0
    nongeocoded_acc_count, nongeocoded_cgs_count, nongeocoded_other_count = 0, 0, 0
    for dataframe in geocoded_dataframe_list:
        geocoded_acc_mask = (dataframe['traffic_type'] == 'accident')
        geocoded_cgs_mask = (dataframe['traffic_type'] == 'congestion')
        geocoded_other_mask = (dataframe['traffic_type'] == 'condition')
        geocoded_acc_data = dataframe.loc[geocoded_acc_mask]
        geocoded_cgs_data = dataframe.loc[geocoded_cgs_mask]
        geocoded_other_data = dataframe.loc[geocoded_other_mask]
        geocoded_acc_count += geocoded_acc_data.shape[0]
        geocoded_cgs_count += geocoded_cgs_data.shape[0]
        geocoded_other_count += geocoded_other_data.shape[0]
    for dataframe in nongeocoded_dataframe_list:
        nongeocoded_acc_mask = (dataframe['traffic_type'] == 'accident')
        nongeocoded_cgs_mask = (dataframe['traffic_type'] == 'congestion')
        nongeocoded_other_mask = (dataframe['traffic_type'] == 'condition')
        non_geocoded_acc_data = dataframe.loc[nongeocoded_acc_mask]
        non_geocoded_cgs_data = dataframe.loc[nongeocoded_cgs_mask]
        non_geocoded_other_data = dataframe.loc[nongeocoded_other_mask]
        nongeocoded_acc_count += non_geocoded_acc_data.shape[0]
        nongeocoded_cgs_count += non_geocoded_cgs_data.shape[0]
        nongeocoded_other_count += non_geocoded_other_data.shape[0]
    print('=' * 20)
    print('In total, we have got {} Weibos'.format(geocoded_acc_count + geocoded_cgs_count +
                                                   geocoded_other_count + nongeocoded_acc_count +
                                                   nongeocoded_cgs_count + nongeocoded_other_count))
    print('Geocoded Accident: {}'.format(geocoded_acc_count))
    print('Geocoded Congestion: {}'.format(geocoded_cgs_count))
    print('Geocoded Other: {}'.format(geocoded_other_count))
    print('Non Geocoded Accident: {}'.format(nongeocoded_acc_count))
    print('Non Geocoded Congestion: {}'.format(nongeocoded_cgs_count))
    print('Non Geocoded Other: {}'.format(nongeocoded_other_count))
    print('=' * 20)


def count_positive_neutral_negative(dataframe, repost_column: str):
    """
    Count the number of positive, neutral and negative Weibos and Reposts
    The code should consider both the Weibo and repost data (if available)
    :param dataframe: a Weibo dataframe containing Weibos and reposts
    :param repost_column: the column name indicating whether this Weibo reposts other weibo
    :return: number of positive, neutral and negative Weibos & reposts
    """
    assert 'sent_weibo' in dataframe, "The dataframe does not contain the sentiment of Weibo"
    assert 'sent_repos' in dataframe, "The dataframe does not contain the sentiment of repost"

    rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text'}
    renamed_data = dataframe.rename(columns=rename_dict).reset_index(drop=True)

    # Use the sentiment of Weibo, not repost
    weibo_dataframe = renamed_data.loc[renamed_data[repost_column] == 'no retweeters']
    repost_dataframe = renamed_data.loc[renamed_data[repost_column] != 'no retweeters']
    pos_count, neutral_count, neg_count = 0, 0, 0
    sent_weibo_counter = Counter(weibo_dataframe['sent_weibo'])
    sent_repost_counter = Counter(repost_dataframe['sent_weibo'])
    pos_count += sent_weibo_counter.get(2, 0)
    pos_count += sent_repost_counter.get(2, 0)
    neutral_count += sent_weibo_counter.get(1, 0)
    neutral_count += sent_repost_counter.get(1, 0)
    neg_count += sent_weibo_counter.get(0, 0)
    neg_count += sent_repost_counter.get(0, 0)
    return pos_count, neutral_count, neg_count


def number_of_tweet_users(dataframe, user_id_column_name, print_value=True):
    """
    Get the number of tweets and number of twitter users
    :param dataframe: the studied dataframe
    :param user_id_column_name: the column name which saves the user ids
    :param print_value: whether print the values or not
    :return: the number of tweets and number of users
    """
    number_of_tweets = dataframe.shape
    number_of_users = len(set(dataframe[user_id_column_name]))
    if print_value:
        print('The number of tweets: {}; The number of unique social media users: {}'.format(
            number_of_tweets, number_of_users))
    else:
        return number_of_tweets, number_of_users


def read_local_file(path, filename, csv_file=True):
    """
    Read a csv or pickle file from a local directory
    :param path: the path which save the local csv file
    :param filename: the studied filename
    :param csv_file: whether this file is a csv file
    :return: a pandas dataframe
    """
    if csv_file:
        dataframe = pd.read_csv(os.path.join(path, filename), encoding='utf-8', index_col=0)
    else:
        dataframe = pd.read_pickle(os.path.join(path, filename))
    return dataframe


def get_edge_embedding_for_mlp(edge_list, embedding_dict, concatenate_or_not=True):
    """
    Get edge embeddings for mlp classifier
    :param edge_list: a python list which contains the edges of a graph
    :param embedding_dict: a dictionary of which key is the node and value is the node2vec embedding
    :param concatenate_or_not: whether we concatenate two node embeddings or not
    :return: the embeddings for edges of a graph
    """
    embeddings = []
    for edge in edge_list:
        node_id1 = edge[0]
        node_id2 = edge[1]
        emb1 = embedding_dict[node_id1]
        emb2 = embedding_dict[node_id2]
        if concatenate_or_not:
            emb_concat = np.concatenate([emb1, emb2], axis=0)
            embeddings.append(emb_concat)
        else:
            emb_multiply = np.multiply(emb1, emb2)
            embeddings.append(emb_multiply)
    return embeddings


def get_network_statistics(g):
    """Get the network statistics of a networkx graph"""
    num_connected_components = nx.number_connected_components(g)
    node_attribute_dict = nx.get_node_attributes(g, 'type')
    edge_attribute_dict = nx.get_edge_attributes(g, 'relation')
    user_dict = {key: value for (key, value) in node_attribute_dict.items() if value == 'user'}
    location_dict = {key: value for (key, value) in node_attribute_dict.items() if value == 'location'}
    user_user_edge = {key: value for (key, value) in edge_attribute_dict.items() if value == 'user_user'}
    location_user_edge = {key: value for (key, value) in edge_attribute_dict.items() if value == 'user_location'}
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    density = nx.density(g)
    avg_clustering_coef = nx.average_clustering(g)
    avg_degree = sum([int(degree[1]) for degree in g.degree()]) / float(num_nodes)
    transitivity = nx.transitivity(g)

    if num_connected_components == 1:
        diameter = nx.diameter(g)
    else:
        diameter = None  # infinite path length between connected components

    network_statistics = {
        'num_connected_components': num_connected_components,
        'num_nodes': num_nodes,
        'num_user_nodes': len(user_dict),
        'num_location_nodes': len(location_dict),
        'num_edges': num_edges,
        'num_user_user_edges': len(user_user_edge),
        'num_user_location_edges': len(location_user_edge),
        'density': density,
        'diameter': diameter,
        'avg_clustering_coef': avg_clustering_coef,
        'avg_degree': avg_degree,
        'transitivity': transitivity
    }

    return network_statistics


def create_labelled_dataframe(dataframe):
    """
    Construct the labeled dataframe
    :param dataframe: a dataframe containing the labeled weibo dataframe
    :return: a final dataframe containing the weibo and reposted weibos
    """
    author_dataframe = dataframe[['weibo_id', 'text', 'label_1']]
    retweeter_dataframe = dataframe[['retweets_id', 'retweeters_text', 'label_2']]

    # Cope with the retweeter dataframe
    retweeter_dataframe_select = retweeter_dataframe.loc[retweeter_dataframe['label_2'] != -1]
    retweeter_dataframe_without_na = retweeter_dataframe_select[~retweeter_dataframe_select['label_2'].isna()]
    retweeter_data_without_duplicates = retweeter_dataframe_without_na.drop_duplicates(subset='retweets_id',
                                                                                       keep='first')
    retweeter_data_without_duplicates['retweets_id'] = retweeter_data_without_duplicates.apply(
        lambda row: row['retweets_id'][1:-1], axis=1)

    final_dataframe = pd.DataFrame(columns=['id', 'text', 'label'])
    author_id_list = list(author_dataframe['weibo_id'])
    author_text_list = list(author_dataframe['text'])
    author_label_list = list(author_dataframe['label_1'])
    author_id_list.extend(list(retweeter_data_without_duplicates['retweets_id']))
    author_text_list.extend(list(retweeter_data_without_duplicates['retweeters_text']))
    author_label_list.extend(list(retweeter_data_without_duplicates['label_2']))
    final_dataframe['id'] = author_id_list
    final_dataframe['text'] = author_text_list
    final_dataframe['label'] = author_label_list

    return final_dataframe


def combine_some_data(path, sample_num: int = None) -> pd.DataFrame:
    """
    Combine some random sampled dataframes from a local path
    :param path: an interested path
    :param sample_num: the number of files we want to consider
    """
    files = os.listdir(path)
    if not sample_num:
        random_sampled_files = files
    else:
        random_sampled_files = sample(files, k=sample_num)

    dataframe_list = []
    for file in random_sampled_files:
        print('Coping with the file: {}'.format(file))
        dataframe = pd.read_csv(os.path.join(path, file), encoding='utf-8', index_col=0)
        dataframe_list.append(dataframe)

    concat_dataframe = pd.concat(dataframe_list, axis=0)
    concat_dataframe_reindex = concat_dataframe.reset_index(drop=True)
    return concat_dataframe_reindex


def get_weibo_from_users(path, user_set):
    """
    Get the Weibo data posted from a set of users, considering both the original post and repost
    :param path: a Weibo data path
    :param user_set: a set of social media users
    :return: a Weibo dataframe posted from a set of users
    """
    official_account_data_list = []
    for file in os.listdir(path):
        print('Coping with the file: {}'.format(file))
        dataframe = pd.read_csv(os.path.join(path, file), encoding='utf-8', index_col=0)
        assert "author_id" in dataframe, "The dataframe should contain a column named 'author_id'"
        assert "retweeters_ids" in dataframe, "The dataframe should contain a column named 'retweeters_ids'"
        decision1 = (dataframe['author_id'].isin(user_set))
        decision2 = (dataframe['retweeters_ids'].isin(user_set))
        dataframe_select = dataframe.loc[decision1 | decision2]
        official_account_data_list.append(dataframe_select)
    concat_data = pd.concat(official_account_data_list, axis=0)
    return concat_data


def get_weibos_from_users_csv(path: str, filename: str, save_path: str, save_filename: str, user_set: set) -> None:
    """
    Get the Weibos posted from users, given a set storing the id of the desired users
    :param path: the path saving the Weibo csv files
    :param filename: the name of the considered file
    :param save_path: the path used to save the Weibos posted by these users in csv format
    :param save_filename: the saved filename
    :param user_set: a set saving the considered users
    :return: None. The result is saved to a local directory specified by the argument "save_path"
    """
    with open(os.path.join(path, filename), 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        counter = 1
        with open(os.path.join(save_path, save_filename), 'w', encoding='utf-8') as new_file:
            field_names = ['author_id', 'weibo_id', 'created_at', 'text', 'lat', 'lon', 'retweeters_ids',
                           'retweeters_text', 'retweets_id']
            csv_writer = csv.DictWriter(new_file, fieldnames=field_names)
            csv_writer.writeheader()
            for line in csv_reader:
                decision1 = line['author_id'] in user_set
                decision2 = line['retweeters_ids'] in user_set
                if decision1 or decision2:
                    print('found {} weibo'.format(counter))
                    csv_writer.writerow(line)


def get_weibos_from_users_json(data_path, json_filename, save_path, user_set):
    """
    Get Weibo data posted from a user id set based on the json files
    :param data_path: path saving the Sina Weibo data
    :param json_filename: the studied json_filename
    :param save_path: path used to save the data
    :param user_set: a set containing the studied user ids
    :return: None
    """
    with open(os.path.join(data_path, json_filename), encoding='utf-8', errors='ignore') as json_file:

        time_list = []
        user_id_list = []
        weibo_id_list = []
        text_list = []
        geo_type_list = []
        lat_list = []
        lon_list = []
        retweeters_id_list = []
        retweets_weibo_id_list = []
        retweeters_text_list = []

        counter = 0

        for line in json_file:
            try:
                data = json.loads(line)
                tweet_list = data['statuses']
                for index, weibo in enumerate(tweet_list):
                    decision1 = weibo['uid'] in user_set
                    try:
                        decision2 = weibo['retweeted_status']['uid'] in user_set
                    except (KeyError, TypeError) as e:
                        decision2 = False

                    if decision1 or decision2:
                        time_list.append(weibo['created_at'])
                        user_id_list.append(weibo['uid'])
                        text_list.append(weibo['text'])
                        weibo_id_list.append(weibo['id'])
                        try:
                            weibo_geo_info = weibo['geo']
                            lat_list.append(str(weibo_geo_info['coordinates'][0]))
                            lon_list.append(str(weibo_geo_info['coordinates'][1]))
                            geo_type_list.append(weibo_geo_info['type'])
                        except (KeyError, TypeError) as e:
                            lat_list.append('Not Given')
                            lon_list.append('Not Given')
                            geo_type_list.append('Not Given')
                        try:
                            retweeted_statuses = weibo['retweeted_status']
                            retweeters_id_list.append(retweeted_statuses['uid'])
                            retweeters_text_list.append(retweeted_statuses['text'])
                            retweets_weibo_id_list.append(retweeted_statuses['id'])
                        except (KeyError, TypeError) as e:
                            retweeters_id_list.append(0)
                            retweeters_text_list.append('no retweeters')
                            retweets_weibo_id_list.append('no retweets')
            except JSONDecodeError as e_1:
                print('The error line is: {}'.format(line))
                print('ignore')
            except TypeError as e_2:
                print('The error line is: {}'.format(line))
                print('ignore')

            if len(weibo_id_list) > 100000:
                counter += 1

                print('{} weibos have been processed!'.format(len(weibo_id_list)))
                result_dataframe = pd.DataFrame(columns=['author_id', 'weibo_id', 'created_at', 'text', 'lat',
                                                         'lon', 'loc_type', 'retweeters_ids',
                                                         'retweeters_text', 'retweets_id'])
                result_dataframe['author_id'] = user_id_list
                result_dataframe['weibo_id'] = weibo_id_list
                result_dataframe['created_at'] = time_list
                result_dataframe['text'] = text_list
                result_dataframe['lat'] = lat_list
                result_dataframe['lon'] = lon_list
                result_dataframe['loc_type'] = geo_type_list
                result_dataframe['retweeters_ids'] = retweeters_id_list
                result_dataframe['retweeters_text'] = retweeters_text_list
                result_dataframe['retweets_id'] = retweets_weibo_id_list
                result_dataframe.to_csv(os.path.join(save_path, json_filename[:9] + '_{}.csv'.format(counter)),
                                        encoding='utf-8')
                # release memory
                user_id_list = []
                weibo_id_list = []
                time_list = []
                text_list = []
                lat_list = []
                lon_list = []
                geo_type_list = []
                retweeters_id_list = []
                retweeters_text_list = []
                retweets_weibo_id_list = []

        result_dataframe = pd.DataFrame(columns=['author_id', 'weibo_id', 'created_at', 'text', 'lat',
                                                 'lon', 'loc_type', 'retweeters_ids',
                                                 'retweeters_text', 'retweets_id'])
        result_dataframe['author_id'] = user_id_list
        result_dataframe['weibo_id'] = weibo_id_list
        result_dataframe['created_at'] = time_list
        result_dataframe['text'] = text_list
        result_dataframe['lat'] = lat_list
        result_dataframe['lon'] = lon_list
        result_dataframe['loc_type'] = geo_type_list
        result_dataframe['retweeters_ids'] = retweeters_id_list
        result_dataframe['retweeters_text'] = retweeters_text_list
        result_dataframe['retweets_id'] = retweets_weibo_id_list
        result_dataframe.to_csv(os.path.join(save_path, json_filename[:9] + '_final.csv'), encoding='utf-8')


def get_traffic_type(dataframe: pd.DataFrame, text_colname: str = 'text', return_counter: bool = False) \
        -> Counter or pd.DataFrame:
    """
    Annotate the Weibos posted by official traffic account with different traffic information type
    :param dataframe: the dataframe saving Weibos posted by official traffic account in Shanghai
    :param text_colname: the name of the column saving the type of traffic-related event with location information
    :param return_counter: print the traffic type directly or not
    :return: dataframe with traffic info type annotation, saved in its 'traffic_type' column
    """
    assert 'text' in dataframe, 'The text column should contain Weibo text posted by official traffic account.'
    traffic_type_list = []
    for index, row in dataframe.iterrows():
        decision1 = any(acc_word in row[text_colname] for acc_word in accident_traffic_word_set)
        decision2 = any(congestion_word in row[text_colname] for congestion_word in congestion_traffic_word_set)
        if decision1 and decision2:  # contain both accident and congestion keywords -> check manually
            traffic_type_list.append('acc-cgs')
        elif decision1 and (not decision2):  # contains one accident keyword but not congestion keyword -> accident
            traffic_type_list.append('accident')
        elif (not decision1) and decision2:  # contains one congestion keyword but not accident keyword -> congestion
            traffic_type_list.append('congestion')
        else:  # Other cases -> a description of traffic condition
            traffic_type_list.append('condition')
    if return_counter:
        traffic_type_counter = Counter(traffic_type_list)
        return traffic_type_counter
    else:
        dataframe_copy = dataframe.copy()
        dataframe_copy['traffic_type'] = traffic_type_list
        return dataframe_copy


def get_weibos_in_given_day(dataframe: pd.DataFrame, check_month: int, check_day: int, save_path: str,
                            save_filename: str) -> None:
    """
    Get Weibos posted in a specific day (Don't consider year)
    :param dataframe: a pandas dataframe saving the Weibo data
    :param check_month: the month we want to check
    :param check_day: the day in a month that we want to check
    :param save_path: the local directory to save the selected data
    :param save_filename: the filename used to save to selected data
    :return: None. The selected data is saved to local directory
    """
    dataframe_copy = dataframe.copy()
    dataframe_copy['local_time'] = dataframe_copy.apply(lambda row: transform_datetime_string_to_datetime(
        row['local_time'], target_timezone=timezone_shanghai), axis=1)
    dataframe_copy['month'] = dataframe_copy.apply(lambda row: row['local_time'].month, axis=1)
    dataframe_copy['day'] = dataframe_copy.apply(lambda row: row['local_time'].day, axis=1)
    dataframe_select = dataframe_copy.loc[
        (dataframe_copy['month'] == check_month) & (dataframe_copy['day'] == check_day)].reset_index(drop=True)
    dataframe_select.to_csv(os.path.join(save_path, save_filename))


def combine_candidate_ids(dataframe: pd.DataFrame) -> set:
    """
    Get the Weibo id set, considering original post and repost
    :param dataframe: a Weibo dataframe
    :return: a Weibo id set
    """
    author_int_set = set(dataframe['weibo_id'])
    retweeter_list = list(dataframe['retweets_id'])
    retweeter_int_set = set(
        [np.int64(str(retweet_id[1:-1])) for retweet_id in retweeter_list if retweet_id != "['no retweets']"])
    # combine the retweet id and author id together
    combine_set = {*author_int_set, *retweeter_int_set}
    return combine_set


def create_cmap_from_gmm_labels(label_list: list, color_list: list):
    """
    Create the colormaps based on the label list
    :param label_list: a label list which map each color given in the color list
    :param color_list: a color list having the colors you want to use. For instance:
    color_list = ['#dc2624', '#2b4750', '#45a0a2', '#e87a59', '#7dcaa9', '#649e7d',  '#dc8018',
    '#c89f91', '#6c6d6c', '#4f6268', '#c7cccf']
    :return: a matplotlib color map that can be used in ax.scatter
    """
    color_iter = itertools.cycle(color_list)
    label_len = len(label_list)
    final_color_list = []
    color_counter = 0
    for color in color_iter:
        if color_counter != label_len - 1:
            final_color_list.append(color)
        else:
            break
        color_counter += 1
    color_maps = LinearSegmentedColormap.from_list('my_list', [Color(c1).rgb for c1 in final_color_list])
    return color_maps


def build_train_data(augment_dataframe: pd.DataFrame, num_for_label: int):
    """
    Build the augmented training dataframe.
    The augmentation process is followed by https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    :param augment_dataframe: the classification dataframe augmented by EDA_Chinese
    :param num_for_label: the number of Weibos for each label
    :return: a pandas dataframe used to train the classification model
    """

    assert 'label_3' in augment_dataframe, "The dataframe should have one column saving label information: label_3"
    train_label_0 = augment_dataframe.loc[augment_dataframe['label_3'] == 0]
    train_label_1 = augment_dataframe.loc[augment_dataframe['label_3'] == 1]
    train_label_2 = augment_dataframe.loc[augment_dataframe['label_3'] == 2]

    train_label_0_select = train_label_0.sample(num_for_label)
    train_label_1_select = train_label_1.sample(num_for_label)
    train_label_2_select = train_label_2.sample(num_for_label)

    concat_data = pd.concat([train_label_0_select, train_label_1_select, train_label_2_select], axis=0)
    concat_data_reindex = concat_data.reset_index(drop=True)

    return concat_data_reindex


def compute_distance_matrix(location_array: np.array) -> np.array:
    """
    Create a symmetric distance matrix based on a location array using geodesic distance
    The geodesic distance is the shortest distance on the surface of an ellipsoidal model of the earth
    :param location_array: a location array, each row records the latitude and longitude of a point
    :return: a numpy array saving the distance between the points
    """
    distance_list = []
    print("In total, we have {} locations.".format(location_array.shape[0]))
    for index, array_first in enumerate(location_array):
        print('Coping with the {}th row'.format(index))
        one_row_result_list = []
        for array_second in location_array:
            if tuple(array_first) == tuple(array_second):
                one_row_result_list.append(0)
            else:
                one_row_result_list.append(geodesic(array_first, array_second).m)
        distance_list.append(one_row_result_list)
    return np.array(distance_list)
