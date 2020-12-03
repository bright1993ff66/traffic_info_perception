import pandas as pd
import os
import data_paths
import numpy as np
from collections import Counter
import statsmodels.api as sm

from data_paths import shapefile_path
from content_analysis.traffic_weibo import accident_traffic_word_set, congestion_traffic_word_set
from content_analysis.sentiment_analysis import output_sentiment_int


def create_hotspot_dataframe(path:str, search_string:str) -> pd.DataFrame:
    """
    Create the Weibo dataframe for one type of traffic hotspots
    :param path: a local path saving the filtered Weibos posted in hotspots
    :param search_string: a string you want to use to find the txt file
    :return: a pandas dataframe containing all the Weibos posted in one type of traffic hotspot
    """
    dataframe_list = []
    for file in os.listdir(path):
        if file.endswith('.csv') and search_string in file:
            dataframe = pd.read_csv(os.path.join(path, file), index_col=0, encoding='utf-8')
            dataframe_list.append(dataframe)
    concat_data = pd.concat(dataframe_list, axis=0)
    select_columns = ['index_val', 'author_id', 'weibo_id', 'created_at', 'text', 'lat', 'lon', 'retweeters',
                      'retweete_1', 'local_time', 'year', 'month', 'traffic_we', 'traffic_re', 'sent_weibo',
                      'sent_repos', 'Name', 'datatype', 'traffic_ty']
    rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text'}
    renamed_concat_data = concat_data[select_columns].rename(columns=rename_dict).reset_index(drop=True)
    return renamed_concat_data


def count_traffic_weibos_in_districts(dataframe, district_name_column):
    """
    Count the number of traffic Weibos across districts
    :param dataframe: a traffic Weibo dataframe, which has a column saving the district location of each row
    :param district_name_column: the column name saving the district name
    :return: the dataframe saving the number of traffic Weibos found in each district
    """
    select_columns = ['index_val', 'author_id', 'weibo_id', 'created_at', 'text', 'lat', 'lon', 'retweeters',
                      'retweete_1', 'local_time', 'year', 'month', 'traffic_we', 'traffic_re', 'sent_weibo',
                      'sent_repos', 'Name', 'datatype', 'traffic_ty']
    rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text'}
    renamed_data = dataframe[select_columns].rename(columns=rename_dict)
    district_counter = Counter(renamed_data[district_name_column])
    district_name_list = list(district_counter.keys())
    result_dataframe = pd.DataFrame(columns=['districts', 'count'])
    result_dataframe['districts'] = district_name_list
    result_dataframe['count'] = [district_counter[name] for name in district_name_list]
    result_dataframe_sorted = result_dataframe.sort_values(by='count', ascending=False).reset_index(drop=True)
    return result_dataframe_sorted


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
              'sent_repos', 'Name']
    rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text'}
    renamed_data = data[select_columns].rename(columns=rename_dict)
    renamed_data_copy = renamed_data.copy()
    renamed_data_copy['datatype'] = [datatype_label]*renamed_data_copy.shape[0]
    # renamed_data_copy['sent_weibo'] = renamed_data_copy.apply(
    #     lambda row: output_sentiment_int(row['sent_weibo']), axis=1)
    # renamed_data_copy['sent_repos'] = renamed_data_copy.apply(
    #     lambda row: output_sentiment_int(row['sent_repos']), axis=1)
    renamed_data_copy.to_csv(os.path.join(data_paths.weibo_data_path, saved_filename), encoding='utf-8')
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
                      'sent_repos', 'Name']
    rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text',
                   'loc_lat': 'lat', 'loc_lon': 'lon'}
    renamed_data = data[select_columns].rename(columns=rename_dict)
    renamed_data_copy = renamed_data.copy()
    renamed_data_copy['datatype'] = [datatype_label]*renamed_data_copy.shape[0]
    # renamed_data_copy['sent_weibo'] = renamed_data_copy.apply(
    #     lambda row: output_sentiment_int(row['sent_weibo']), axis=1)
    # renamed_data_copy['sent_repos'] = renamed_data_copy.apply(
    #     lambda row: output_sentiment_int(row['sent_repos']), axis=1)
    renamed_data_copy.to_csv(os.path.join(data_paths.weibo_data_path, saved_filename), encoding='utf-8')
    return renamed_data_copy


def get_traffic_dataframe_statistics(dataframe):
    """
    Count different type of Weibo dataframe
    :param dataframe: a Weibo dataframe
    :return:
    """
    assert 'datatype' in dataframe, 'Make sure we have a column saving the type of the dataframe'
    dataframe_geocoded = dataframe.loc[dataframe['datatype'] == 'geocoded']
    geocoded_weibo = dataframe_geocoded.loc[dataframe_geocoded['retweeters_text'] == 'no retweeters']
    geocoded_repost = dataframe_geocoded.loc[dataframe_geocoded['retweeters_text'] != 'no retweeters']
    # geocoded_repost_without_duplicates = geocoded_repost.drop_duplicates(subset=['retweeters'])
    dataframe_not_geocoded = dataframe.loc[dataframe['datatype'] != 'geocoded']
    nongeocoded_weibo = dataframe_not_geocoded.loc[dataframe_not_geocoded['retweeters_text'] == 'no retweeters']
    nongeocoded_repost = dataframe_not_geocoded.loc[dataframe_not_geocoded['retweeters_text'] != 'no retweeters']
    # nongeocoded_repost_without_duplicates = nongeocoded_repost.drop_duplicates(subset=['retweeters'])
    geo_weibo_traf_list, geo_repost_traf_list = list(geocoded_weibo['traffic_weibo']), \
                                                list(geocoded_repost['traffic_repost'])
    not_geo_weibo_traf_list, not_geo_repost_traf_list = list(nongeocoded_weibo['traffic_weibo']), \
                                                list(nongeocoded_repost['traffic_repost'])
    counter_geo_weibo, counter_geo_repost = Counter(geo_weibo_traf_list), Counter(geo_repost_traf_list)
    counter_not_geo_weibo, counter_not_geo_repost = Counter(not_geo_weibo_traf_list), Counter(not_geo_repost_traf_list)
    return [counter_geo_weibo[2], counter_geo_weibo[1], counter_geo_repost[2], counter_geo_repost[1]], [
        counter_not_geo_weibo[2], counter_not_geo_weibo[1], counter_not_geo_repost[2], counter_not_geo_repost[1]]


def count_traffic_in_dataframe(dataframe):
    """
    Count the number of Weibos in each type
    :param dataframe: a Weibo dataframe
    :return:
    """
    traffic_geo_count = 0
    traffic_geo_have_loc = 0
    traffic_geo_not_have_loc = 0
    traffic_non_geo_count = 0
    traffic_non_geo_have_loc = 0
    traffic_non_geo_not_have_loc = 0

    geo_list, not_geo_list = get_traffic_dataframe_statistics(dataframe)
    traffic_geo_count += sum(geo_list)
    traffic_geo_have_loc += geo_list[0]
    traffic_geo_have_loc += geo_list[2]
    traffic_geo_not_have_loc += geo_list[1]
    traffic_geo_not_have_loc += geo_list[3]
    traffic_non_geo_count += sum(not_geo_list)
    traffic_non_geo_have_loc += not_geo_list[0]
    traffic_non_geo_have_loc += not_geo_list[2]
    traffic_non_geo_not_have_loc += not_geo_list[1]
    traffic_non_geo_not_have_loc += not_geo_list[3]

    print('In total, we have {} traffic relevant Weibos'.format(traffic_geo_count+ traffic_non_geo_count))
    print('We have got {} traffic Weibos with geo information'.format(traffic_geo_count))
    print('\tAmong geocoded traffic Weibos, we have {} Weibos having loc info in text and {} Weibos not '
          'having loc info in text'.format(traffic_geo_have_loc, traffic_geo_not_have_loc))
    print('We have got {} traffic Weibos without geo-information'.format(traffic_non_geo_count))
    print('\tAmong not geocoded traffic Weibos, we have {} Weibos having loc info and {} Weibos not '
          'having loc info'.format(traffic_non_geo_have_loc, traffic_non_geo_not_have_loc))


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
    count_traffic_in_dataframe(combined_traffic_shanghai_copy)
    print('Derive the traffic information type...')
    # Get the traffic information type based on keywords
    # If a weibo contains only congestion keywords, we regard this Weibo as congestion-related Weibo
    # If a weibo contains accident-related keywords, we regard this Weibo as accident-related Weibo
    traffic_type_list = []
    for index, row in combined_traffic_shanghai_copy.iterrows():
        traffic_type = ''
        if row['datatype'] != 'nongeo_repost': # geocoded data and nongeocoded weibo
            considered_text = row['text']
        else: # nongeocoded repost
            considered_text = row['retweeters_text']
        decision_conges = any(traffic_word in considered_text for traffic_word in congestion_traffic_word_set)
        decision_accident = any(traffic_word in considered_text for traffic_word in accident_traffic_word_set)
        if decision_conges and (not decision_accident):
            traffic_type = 'congestion'
        if decision_accident:
            traffic_type = 'accident'
        if (not decision_conges) and (not decision_accident):
            traffic_type = 'condition'
        traffic_type_list.append(traffic_type)
    combined_traffic_shanghai_copy['traffic_type'] = traffic_type_list
    combined_traffic_shanghai_copy['lat'] = combined_traffic_shanghai_copy['lat'].astype(np.float64)
    combined_traffic_shanghai_copy['lon'] = combined_traffic_shanghai_copy['lon'].astype(np.float64)
    combined_traffic_shanghai_copy.to_csv(os.path.join(data_paths.weibo_data_path,
                                                       'combined_traffic_weibo_shanghai.csv'), encoding='utf-8')
    return combined_traffic_shanghai_copy


def create_official_fishnet_dataframe(data: pd.DataFrame, considered_column: str) -> pd.DataFrame:
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
    result_dataframe_reindex = result_dataframe_sorted.reset_index(drop=True)
    return result_dataframe_reindex


def create_weibo_fishnet_dataframe(data: pd.DataFrame, considered_column: str) -> pd.DataFrame:
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


def create_data_for_regres(official_traffic_data, traffic_weibo_data, considered_count: int):
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
    merged_data_select_sorted = merged_data_select.sort_values(by='count', ascending=False)
    merged_data_select_reindex = merged_data_select_sorted.reset_index(drop=True)
    return merged_data_select_reindex


def regres_analysis(dataframe: pd.DataFrame, feature_columns: list, predict_column: list, output_dataframe: bool):
    """
    Conduct the regression analysis given dataframe containing features and y values
    :param dataframe: the dataframe for regression analysis
    :param feature_columns: the columns saving the independent variables
    :param predict_column: the column saving the dependent variable
    :param output_dataframe: whether we save the pandas dataframe regression result to local
    """
    feature_values = dataframe[feature_columns].values
    y_values = dataframe[predict_column]
    mod = sm.OLS(y_values, feature_values)
    res = mod.fit()
    if output_dataframe:
        res_summary = res.summary(xname=['# of Accidents', '# of Congestions', '# of Conditions', 'Sentiment Index'])
        results_as_html = res_summary.tables[1].as_html()
        regression_result = pd.read_html(results_as_html, header=0, index_col=0)[0]
        regression_result.to_csv(os.path.join(data_paths.weibo_data_path, 'regression_result.csv'), encoding='utf-8')
    else:
        print(res.summary())


if __name__ == '__main__':
    # Count the number of traffic Weibos across districts
    combined_dataframe = pd.read_csv(os.path.join(data_paths.district_analysis_join_result,
                                                  'Weibo_shanghai_across_districts.txt'), encoding='utf-8', index_col=0)
    district_count = count_traffic_weibos_in_districts(combined_dataframe, district_name_column='Name')
    district_count.to_csv(os.path.join(data_paths.weibo_data_path, 'district_count.csv'), encoding='utf-8')
    # Get the Weibos posted in accident hotspot and congestion hotspot
    acc_hotspot_weibo = create_hotspot_dataframe(path=data_paths.hotspot_text_path, search_string='acc')
    conges_hotspot_weibo = create_hotspot_dataframe(path=data_paths.hotspot_text_path, search_string='cgs')
    acc_hotspot_weibo.to_csv(os.path.join(data_paths.hotspot_text_path, 'weibo_acc_hotspot_with_sent.csv'),
                             encoding='utf-8')
    conges_hotspot_weibo.to_csv(os.path.join(data_paths.hotspot_text_path, 'weibo_cgs_hotspot_with_sent.csv'),
                                encoding='utf-8')