import os
import pandas as pd
import numpy as np
from collections import Counter
import itertools
import networkx as nx
from colour import Color
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

import re
from random import sample
import pytz
from datetime import datetime

# Specify the timezone in Shanghai
timezone_shanghai = pytz.timezone('Asia/Shanghai')

# Specify the random seed
random_seed = 7
np.random.seed(random_seed)

# Set the standard scaler
scaler = StandardScaler()


def delete_user(text):
    """
    Delete the @user in the weibo or tweet text
    :param text: a weibo or tweet string
    :return: a text string without @user
    """
    result_text = re.sub("@[^，，：：\s@:]+", "", text)
    return result_text


def merge_dict(sum_dict, a_dict):
    """
    Merge a sum dictionary and a dictionary for a csv file
    Args:
        sum_dict: the sum_dict records the total number of tweets found in one city
        a_dict: a count dict for a csv tweet file
    Returns: a sum_dict which has added values from a_dict
    """
    if a_dict == Counter(): return sum_dict
    for key in a_dict:
        sum_dict[key] += a_dict[key]
    return sum_dict


def normalize_data(dataframe, select_columns: list, density_threshold: int):
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


def get_dataframe_in_gmm_clusters(dataframe, cluster_id_dict, save_path, traffic_event_type: str):
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
        cluster_dataframe.to_csv(os.path.join(save_path, traffic_event_type+'_gmm_{}.csv'.format(cluster_name)))


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
    :param dataframe: a traffic Weibo dataframe
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
    print('In total, we have got {} Weibos'.format(geocoded_acc_count+geocoded_cgs_count+
                                                   geocoded_other_count+nongeocoded_acc_count+
                                                   nongeocoded_cgs_count+nongeocoded_other_count))
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

    # repost_dataframe_weibo = renamed_data.loc[renamed_data[repost_column] != 'no retweeters']  # Get all the reposts
    # repost_dataframe_repost = repost_dataframe_weibo.drop_duplicates(subset=['retweeters_text'])
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
    embs = []
    for edge in edge_list:
        node_id1 = edge[0]
        node_id2 = edge[1]
        emb1 = embedding_dict[node_id1]
        emb2 = embedding_dict[node_id2]
        if concatenate_or_not:
            emb_concat = np.concatenate([emb1, emb2], axis=0)
            embs.append(emb_concat)
        else:
            emb_multiply = np.multiply(emb1, emb2)
            embs.append(emb_multiply)
    return embs


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
    """Combine some random sampled dataframes from a local path"""
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


def transform_string_time_to_datetime(time_string, target_time_zone, convert_utc_time=False):
    """
    Transform the string time to the datetime
    :param time_string: a time string
    :param target_time_zone: the target time zone
    :param convert_utc_time: whether transfer the datetime object to utc first. This is true when the
    time string is recorded as the UTC time
    :return:
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


def encode_time(time_string):
    """
    Encode the time string in the official traffic accident dataframe
    :param time_string:
    :param target_timezone:
    :return:
    """
    date_str, time_str = time_string.split(' ')
    year, month, day = date_str.split('/')
    hour, minute = time_str.split(':')
    return datetime(int(year), int(month), int(day), int(hour), int(minute), 0)


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
        (dataframe_copy['month'] == check_month) & (dataframe_copy['day'] == check_day)]
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
    :param color_list: a color list having the colors you want to use. For instance: color_list = ['#dc2624',
    '#2b4750', '#45a0a2', '#e87a59', '#7dcaa9', '#649e7d',  '#dc8018', '#c89f91', '#6c6d6c', '#4f6268', '#c7cccf']
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
