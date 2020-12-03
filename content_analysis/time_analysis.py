import pandas as pd
import os
import pytz
from collections import Counter

import data_paths
from content_analysis.location_analysis import get_kde_weibo_for_arcmap
from visualizations import create_hour_weekday_plot

# Specify the timezone in Shanghai
timezone_shanghai = pytz.timezone('Asia/Shanghai')

# Load the dataframes
geocoded_weibo_shanghai = pd.read_csv(os.path.join(data_paths.district_analysis_join_result,
                                                   'geo_traffic_shanghai.csv'), encoding='utf-8')
nongeocoded_weibo_traffic_shanghai = pd.read_csv(os.path.join(
    data_paths.district_analysis_join_result, 'nongeo_weibo_traffic_shanghai.csv'), encoding='utf-8')
nongeocoded_repost_traffic_shanghai = pd.read_csv(os.path.join(
    data_paths.district_analysis_join_result, 'nongeo_repost_traffic_shanghai.csv'), encoding='utf-8')
accident_hotspot = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'weibo_acc_hotspot_with_sent.csv'),
                               encoding='utf-8', index_col=0)
conges_hotspot = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'weibo_cgs_hotspot_with_sent.csv'),
                             encoding='utf-8', index_col=0)

# # Clean the weibo data
# combined_traffic_shanghai = get_kde_weibo_for_arcmap(geo_data=geocoded_weibo_shanghai,
#                                                      nongeo_weibo_data=nongeocoded_weibo_traffic_shanghai,
#                                                      nongeo_repost_data=nongeocoded_repost_traffic_shanghai)
# create_hour_weekday_plot(combined_traffic_shanghai, color_hour='#FB8072', color_weekday='#80B1D3',
#                                         hour_save_filename='traffic_hour.png',
#                                         weekday_save_filename='traffic_weekday.png',
#                          title_hour='Traffic-related Weibos in Shanghai by Hour',
#                          title_weekday='Traffic-related Weibos in Shanghai by Weekday')
create_hour_weekday_plot(conges_hotspot, color_hour='#FB8072', color_weekday='#80B1D3',
                                        hour_save_filename='traffic_conges_hotspot_hour.png',
                                        weekday_save_filename='traffic_conges_hotspot_weekday',
                         title_hour='Congestion-related Weibos in Shanghai by Hour',
                         title_weekday='Congestion-related Weibos in Shanghai by Weekday')
create_hour_weekday_plot(accident_hotspot, color_hour='#FB8072', color_weekday='#80B1D3',
                                        hour_save_filename='traffic_acc_hotspot_hour.png',
                                        weekday_save_filename='traffic_acc_hotspot_weekday.png',
                         title_hour='Accident-related Weibos in Shanghai by Hour',
                         title_weekday='Accident-related Weibos in Shanghai by Weekday')
