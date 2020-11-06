import pandas as pd
import os
import pytz

import data_paths
from content_analysis.traffic_weibo import accident_traffic_word_set, congestion_traffic_word_set
import visualizations

# Specify the timezone in Shanghai
timezone_shanghai = pytz.timezone('Asia/Shanghai')


def process_geocoded_data(data:pd.DataFrame, datatype_label:str, saved_filename: str) -> pd.DataFrame:
    """
    Get the cleaned geocoded data after ArcMap process
    :param data: a geocoded Weibo dataframe after ArcMap processing
    :param datatype_label: a string label some of the datatype
    :param saved_filename: the filename used to save to the local directory
    :return: a cleaned geocoded dataframe
    """
    select_columns = ['index_val', 'author_id', 'weibo_id', 'created_at', 'text', 'lat', 'lon', 'retweeters',
               'retweete_1', 'local_time', 'year', 'month', 'traffic_we', 'traffic_re', 'sent_weibo',
              'sent_repos', 'Name', 'Shape_Area']
    rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text'}
    renamed_data = data[select_columns].rename(columns=rename_dict)
    renamed_data_copy = renamed_data.copy()
    renamed_data_copy['datatype'] = [datatype_label]*renamed_data_copy.shape[0]
    renamed_data.to_csv(os.path.join(data_paths.weibo_data_path, saved_filename), encoding='utf-8')
    return renamed_data_copy


def process_nongeocoded_data(data:pd.DataFrame, datatype_label:str, saved_filename:str) -> pd.DataFrame:
    """
    Get the cleaned nongeocoded data after ArcMap process
    :param data: a nongeocoded Weibo dataframe after ArcMap processing
    :param datatype_label: a string label some of the datatype
    :param saved_filename: the filename used to save to the local directory
    :return: a cleaned nongeocoded dataframe
    """
    select_columns = ['index_val', 'author_id', 'weibo_id', 'created_at', 'text', 'loc_lat', 'loc_lon', 'retweeters',
               'retweete_1', 'local_time', 'year', 'month', 'traffic_we', 'traffic_re', 'sent_weibo',
              'sent_repos', 'Name', 'Shape_Area']
    rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text', 'loc_lat': 'lat',
                  'loc_lon': 'lon'}
    renamed_data = data[select_columns].rename(columns=rename_dict)
    renamed_data_copy = renamed_data.copy()
    renamed_data_copy['datatype'] = [datatype_label]*renamed_data_copy.shape[0]
    renamed_data.to_csv(os.path.join(data_paths.weibo_data_path, saved_filename), encoding='utf-8')
    return renamed_data_copy


def get_kde_weibo_for_arcmap(geo_data, nongeo_weibo_data, nongeo_repost_data) -> pd.DataFrame:
    """
    Get the combined weibo dataframe for social media based kde processing (find traffic hotspots)
    :param geo_data: the Weibo data with geo information (latitude, longitude)
    :param nongeo_weibo_data: the nongeocoded traffic Weibo data
    :param nongeo_repost_data: the nongeocoded traffic Repost data
    :return: a combined traffic Weibo dataframe
    """
    processed_geo = process_geocoded_data(geo_data, datatype_label='geocoded',
                                          saved_filename='geocoded_traffic_shanghai.csv')
    processed_nongeo_weibo = process_nongeocoded_data(nongeo_weibo_data, datatype_label='nongeo_weibo',
                                                      saved_filename='nongeocoded_weibo_traffic_shanghai.csv')
    processed_nongeo_repost = process_nongeocoded_data(nongeo_repost_data, datatype_label='nongeo_repost',
                                                       saved_filename='nongeocoded_repost_traffic_shanghai.csv')
    combined_traffic_shanghai = pd.concat([processed_geo, processed_nongeo_weibo, processed_nongeo_repost], axis=0)
    combined_traffic_shanghai_copy = combined_traffic_shanghai.copy()
    # Get the traffic information type based on keywords
    # If a weibo contains only congestion keywords, we regard this Weibo as congestion-related Weibo
    # If a weibo contains accident-related keywords, we regard this Weibo as accident-related Weibo
    traffic_type_list = []
    for index, row in combined_traffic_shanghai_copy.iterrows():
        traffic_type = ''
        if row['datatype'] != 'nongeo_repost':
            considered_text = row['text']
        else:
            considered_text = row['retweeters_text']
        decision1 = any(traffic_word in considered_text for traffic_word in congestion_traffic_word_set)
        decision2 = any(traffic_word in considered_text for traffic_word in accident_traffic_word_set)
        if decision1 and (not decision2):
            traffic_type = 'congestion'
        if decision2:
            traffic_type = 'accident'
        if (not decision1) and (not decision2):
            traffic_type = 'condition'
        traffic_type_list.append(traffic_type)
    combined_traffic_shanghai_copy['traffic_type'] = traffic_type_list
    combined_traffic_shanghai_copy.to_csv(os.path.join(data_paths.weibo_data_path,
                                                       'combined_traffic_weibo_shanghai.csv'), encoding='utf-8')
    print('Total number of traffic-related messages is: {}'.format(combined_traffic_shanghai_copy.shape[0]))
    return combined_traffic_shanghai_copy


if __name__ == '__main__':
    geocoded_weibo_shanghai = pd.read_csv(os.path.join(data_paths.shapefile_path, 'geo_traffic_shanghai.txt'),
                                          encoding='utf-8')
    nongeocoded_weibo_traffic_shanghai = pd.read_csv(os.path.join(
        data_paths.shapefile_path, 'nongeo_weibo_traffic_shanghai.txt'), encoding='utf-8')
    nongeocoded_repost_traffic_shanghai = pd.read_csv(os.path.join(
        data_paths.shapefile_path, 'nongeo_repost_traffic_shanghai.txt'), encoding='utf-8')

    # Clean the weibo data
    combined_traffic_shanghai = get_kde_weibo_for_arcmap(geo_data=geocoded_weibo_shanghai,
                                                         nongeo_weibo_data=nongeocoded_weibo_traffic_shanghai,
                                                         nongeo_repost_data=nongeocoded_repost_traffic_shanghai)
    visualizations.create_hour_weekday_plot(combined_traffic_shanghai, color_hour='#FB8072', color_weekday='#80B1D3')