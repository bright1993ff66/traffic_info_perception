import pandas as pd
import os
import data_paths
from collections import Counter


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


if __name__ =='__main__':
    combined_dataframe = pd.read_csv(os.path.join(data_paths.weibo_data_path, 'combined_traffic_weibo_shanghai.csv'),
                                     encoding='utf-8', index_col=0)
    district_count = count_traffic_weibos_in_districts(combined_dataframe, district_name_column='Name')
    district_count.to_csv(os.path.join(data_paths.weibo_data_path, 'district_count.csv'), encoding='utf-8')
