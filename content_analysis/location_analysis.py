import pandas as pd
import os
import data_paths
import numpy as np
from collections import Counter

from data_paths import shapefile_path
from visualizations import correlation_plot


def count_traffic_weibos_in_districts(dataframe, district_name_column):
    """
    Count the number of traffic Weibos across districts
    :param dataframe: a traffic Weibo dataframe, which has a column saving the district location of each row
    :param district_name_column: the column name saving the district name
    :return: the dataframe saving the number of traffic Weibos found in each district
    """
    district_counter = Counter(dataframe[district_name_column])
    district_name_list = list(district_counter.keys())
    result_dataframe = pd.DataFrame(columns=['districts', 'count'])
    result_dataframe['districts'] = district_name_list
    result_dataframe['count'] = [district_counter[name] for name in district_name_list]
    result_dataframe_sorted = result_dataframe.sort_values(by='count', ascending=False).reset_index(drop=True)
    return result_dataframe_sorted


def create_official_fishnet_dataframe(data:pd.DataFrame, considered_column):
    """
    Count the number of real-world traffic events in each fishnet cell
    :param data: dataframe saving the location of real-world traffic event in which fishnet cell
    :param considered_column: the column saving the fishnet cell num that a real-world traffic event belongs to
    :return: dataframe storing the number of real-world traffic events in each fishnet cell
    """
    cell_counter = Counter(data[considered_column])
    cell_list = list(cell_counter.keys())
    value_list = [cell_counter[cell_key] for cell_key in cell_list]
    result_dataframe = pd.DataFrame(columns=['cell_num', 'count'])
    result_dataframe['cell_num'] = cell_list
    result_dataframe['count'] = value_list
    result_dataframe_sorted = result_dataframe.sort_values(by='count', ascending=False)
    result_dataframe_reindex = result_dataframe_sorted.reset_index(drop = True)
    return result_dataframe_reindex


def create_weibo_fishnet_dataframe(data:pd.DataFrame, considered_column):
    """
    Count the number of traffic Weibos in each fishnet cell
    :param data: dataframe saving the location of traffic Weibo in which fishnet cell
    :param considered_column: the column saving the fishnet cell num that a Weibo belongs to
    :return: dataframe storing the number of traffic Weibos in each fishnet cell
    """
    if 'retweeters_text' not in data:
        data_renamed = data.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        data_renamed = data.copy()
    cell_counter = Counter(data_renamed[considered_column])
    cell_list = list(cell_counter.keys())
    accident_count_list, congestion_count_list, condition_count_list = [], [], []
    pos_sent_list, neg_sent_list = [], []
    for cell in cell_list:
        cell_data = data_renamed.loc[data_renamed[considered_column] == cell]
        traffic_type_counter = Counter(cell_data['traffic_ty'])
        accident_count_list.append(traffic_type_counter['accident'])
        congestion_count_list.append(traffic_type_counter['congestion'])
        condition_count_list.append(traffic_type_counter['condition'])
        sent_result, sent_counter = [], Counter()
        for index, row in cell_data.iterrows():
            weibo_sent = eval(row['sent_weibo'])[0]
            sent_result.append(weibo_sent)
            if row['retweeters_text'] != 'no retweeters':
                repost_sent = eval(row['sent_repos'])[0]
                sent_result.append(repost_sent)
        sent_counter = Counter(sent_result)
        pos_sent_list.append(sent_counter['positive'])
        neg_sent_list.append(sent_counter['negative'])
    sent_index_array = (np.array(neg_sent_list) - np.array(pos_sent_list)) / (
                np.array(neg_sent_list) + np.array(pos_sent_list))
    result_dataframe = pd.DataFrame(columns=['cell_num', 'acc_count', 'conges_count', 'condition_count', 'sent_index'])
    result_dataframe['cell_num'] = cell_list
    result_dataframe['acc_count'] = accident_count_list
    result_dataframe['conges_count'] = congestion_count_list
    result_dataframe['condition_count'] = condition_count_list
    result_dataframe['sent_index'] = sent_index_array
    result_dataframe['total'] = result_dataframe.apply(
        lambda row: row['acc_count'] + row['conges_count'] + row['condition_count'], axis=1)
    result_dataframe_sorted = result_dataframe.sort_values(by='total', ascending=False)
    result_dataframe_reindex = result_dataframe_sorted.reset_index(drop=True)
    return result_dataframe_reindex


def create_data_for_regres(official_traffic_data, traffic_weibo_data, considered_count:int):
    """
    Create the data for regression based on the spatial join result based on traffic data from official traffic
    account and traffic relevant Weibo
    :param official_traffic_data: dataframe saving the official traffic data spatial joined on fishnet data
    :param traffic_weibo_data: dataframe saving the traffic relevant data spatial joined on fishnet data
    :param considered_count: the least number of real-world traffic events in a fishnet cell
    :return: dataframe for regression
    """
    considered_cells = set(official_traffic_data['FID_shan_1'])
    traffic_weibo_fishnet_select = traffic_weibo_data.loc[traffic_weibo_data['FID_2'].isin(considered_cells)]
    weibo_fishnet_count = create_weibo_fishnet_dataframe(traffic_weibo_fishnet_select, considered_column='FID_2')
    official_fishnet_count = create_official_fishnet_dataframe(official_traffic_data, considered_column='FID_shan_1')
    assert 'cell_num' in official_fishnet_count and 'cell_num' in weibo_fishnet_count, \
        'The fishnet cell column is not prepared!'
    merged_data = pd.merge(weibo_fishnet_count, official_fishnet_count, how='inner', on=['cell_num'])
    merged_data_select = merged_data.loc[merged_data['count'] >= considered_count]
    merged_data_select_sorted = merged_data_select.sort_values(by = 'count', ascending=False)
    merged_data_select_reindex = merged_data_select_sorted.reset_index(drop=True)
    return merged_data_select_reindex


if __name__ =='__main__':
    combined_dataframe = pd.read_csv(os.path.join(data_paths.weibo_data_path, 'combined_traffic_weibo_shanghai.csv'),
                                     encoding='utf-8', index_col=0)
    district_count = count_traffic_weibos_in_districts(combined_dataframe, district_name_column='Name')
    district_count.to_csv(os.path.join(data_paths.weibo_data_path, 'district_count.csv'), encoding='utf-8')

    traffic_weibo_fishnet = pd.read_csv(os.path.join(shapefile_path, 'traffic_weibo_fishnet50.txt'), encoding='utf-8',
                                        index_col=0)
    official_fishnet = pd.read_csv(os.path.join(shapefile_path, 'traffic_official_weibo_fishnet50.txt'),
                                   encoding='utf-8', index_col=0)
    concat_data_for_regres = create_data_for_regres(official_traffic_data=official_fishnet,
                                                    traffic_weibo_data=traffic_weibo_fishnet,
                                                    considered_count=30)
    print(concat_data_for_regres.head())
    correlation_plot(concat_data_for_regres, considered_column_list=['acc_count', 'conges_count',
                                                                     'condition_count', 'sent_index', 'count'],
                     save_filename='total_weibo_sentiment_official_count_correlation.png')
