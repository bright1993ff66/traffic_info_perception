import pandas as pd
import os
import pytz

import data_paths
from content_analysis.location_analysis import get_kde_weibo_for_arcmap
from visualizations import create_day_plot, create_hour_weekday_plot, create_hour_weekday_with_stack_sentiment, \
    hotspot_day_plot, hotspot_actual_day_plot

# Specify the timezone in Shanghai
timezone_shanghai = pytz.timezone('Asia/Shanghai')

# Load the Weibo dataframes
geocoded_weibo_shanghai = pd.read_csv(os.path.join(data_paths.district_analysis_join_result,
                                                   'geo_traffic_shanghai.csv'), encoding='utf-8')
nongeocoded_weibo_traffic_shanghai = pd.read_csv(os.path.join(
    data_paths.district_analysis_join_result, 'nongeo_weibo_traffic_shanghai.csv'), encoding='utf-8')
nongeocoded_repost_traffic_shanghai = pd.read_csv(os.path.join(
    data_paths.district_analysis_join_result, 'nongeo_repost_traffic_shanghai.csv'), encoding='utf-8')
accident_hotspot = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'weibo_acc_hotspot_with_sent.txt'),
                               encoding='utf-8', index_col=0)
conges_hotspot = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'weibo_cgs_hotspot_with_sent.txt'),
                             encoding='utf-8', index_col=0)
acc_hotspot_4 = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'acc_hotspot_4.txt'),
                                encoding='utf-8', index_col=0)
acc_hotspot_7 = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'acc_hotspot_7.txt'),
                                encoding='utf-8', index_col=0)
acc_hotspot_11 = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'acc_hotspot_11.txt'),
                                encoding='utf-8', index_col=0)
acc_hotspot_19 = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'acc_hotspot_19.txt'),
                                encoding='utf-8', index_col=0)
cgs_hotspot_1 = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'cgs_hotspot_1.txt'),
                                encoding='utf-8', index_col=0)
cgs_hotspot_3 = pd.read_csv(os.path.join(data_paths.hotspot_text_path, 'cgs_hotspot_3.txt'),
                                encoding='utf-8', index_col=0)
# Load the actual traffic records
official_acc_shanghai = pd.read_csv(os.path.join(data_paths.weibo_data_path, 'official_accident_shanghai.csv'),
                                    index_col=0, encoding='utf-8')
official_cgs_shanghai = pd.read_csv(os.path.join(data_paths.weibo_data_path, 'official_congestion_shanghai.csv'),
                                    index_col=0, encoding='utf-8')

# Get the combined Weibo data
combined_traffic_shanghai = get_kde_weibo_for_arcmap(geo_data=geocoded_weibo_shanghai,
                                                     nongeo_weibo_data=nongeocoded_weibo_traffic_shanghai,
                                                     nongeo_repost_data=nongeocoded_repost_traffic_shanghai)

# For all the traffic relevant Weibos
create_day_plot(combined_traffic_shanghai, title='Daily Posts of Weibos', save_filename='traffic_dayfigure.png',
                set_percentile=99)
create_hour_weekday_plot(combined_traffic_shanghai, color_hour='#FB8072', color_weekday='#80B1D3',
                         hour_save_filename='traffic_hour.png',
                         weekday_save_filename='traffic_weekday.png',
                         title_hour='Traffic-related Weibos in Shanghai by Hour',
                         title_weekday='Traffic-related Weibos in Shanghai by Weekday')

# For the accident hotspot
create_hour_weekday_plot(accident_hotspot, color_hour='#FB8072', color_weekday='#80B1D3',
                         hour_save_filename='traffic_acc_hotspot_hour.png',
                         weekday_save_filename='traffic_acc_hotspot_weekday.png',
                         title_hour='Accident-related Weibos in Accident Hotspot by Hour',
                         title_weekday='Accident-related Weibos in Accident Hotspot by Weekday')
create_hour_weekday_with_stack_sentiment(accident_hotspot, color_neg='#FA6466', color_neutral='#FAF838',
                                         color_pos='#4BB8FA',
                                         hour_save_filename='traffic_acc_hotspot_hour_with_sent.png',
                                         weekday_save_filename='traffic_acc_hotspot_weekday_with_sent.png',
                                         title_hour='Accident-related Weibos in Accident Hotspot by Hour',
                                         title_weekday='Accident-related Weibos in Accident Hotspot by Weekday')
# Day plot for the accident hotspot with different density class
hotspot_day_plot(hotspot_dataframe=accident_hotspot, title='Accident-related Weibos in Accident Hotspot by Day',
                 save_filename='traffic_acc_hotspot_per_day.png', set_percentile=97.5, color_pos='#4BB8FA',
                 color_neutral='#FAF838', color_neg='#FA6466')
hotspot_day_plot(hotspot_dataframe=acc_hotspot_4,
                 title='Accident-related Weibos in Accident Hotspot by Day\n(Accident Hotspot ID = 4)',
                 save_filename='traffic_acc_hotspot_per_day_hotspot_4.png', set_percentile=97.5,
                 color_pos='#4BB8FA',
                 color_neutral='#FAF838', color_neg='#FA6466')
hotspot_day_plot(hotspot_dataframe=acc_hotspot_7,
                 title='Accident-related Weibos in Accident Hotspot by Day\n(Accident Hotspot ID = 7)',
                 save_filename='traffic_acc_hotspot_per_day_hotspot_7.png', set_percentile=97.5,
                 color_pos='#4BB8FA',
                 color_neutral='#FAF838', color_neg='#FA6466')
hotspot_day_plot(hotspot_dataframe=acc_hotspot_11,
                 title='Accident-related Weibos in Accident Hotspot by Day\n(Accident Hotspot ID = 11)',
                 save_filename='traffic_acc_hotspot_per_day_hotspot_11.png', set_percentile=97.5,
                 color_pos='#4BB8FA',
                 color_neutral='#FAF838', color_neg='#FA6466')
hotspot_day_plot(hotspot_dataframe=acc_hotspot_19,
                 title='Accident-related Weibos in Accident Hotspot by Day\n(Accident Hotspot ID = 19)',
                 save_filename='traffic_acc_hotspot_per_day_hotspot_19.png', set_percentile=97.5,
                 color_pos='#4BB8FA',
                 color_neutral='#FAF838', color_neg='#FA6466')

# For the congestion hotspot
create_hour_weekday_plot(conges_hotspot, color_hour='#FB8072', color_weekday='#80B1D3',
                                        hour_save_filename='traffic_conges_hotspot_hour.png',
                                        weekday_save_filename='traffic_conges_hotspot_weekday',
                         title_hour='Congestion-related Weibos in Congestion Hotspot by Hour',
                         title_weekday='Congestion-related Weibos in Congestion Hotspot by Weekday')
create_hour_weekday_with_stack_sentiment(conges_hotspot, color_neg='#FA6466', color_neutral='#FAF838',
                                         color_pos='#4BB8FA',
                                         hour_save_filename='traffic_conges_hotspot_hour_with_sent.png',
                                         weekday_save_filename='traffic_conges_hotspot_weekday_with_sent.png',
                                         title_hour='Congestion-related Weibos in Congestion Hotspot by Hour',
                                         title_weekday='Congestion-related Weibos in Congestion Hotspot by Weekday')
hotspot_day_plot(hotspot_dataframe=conges_hotspot, title='Congestion-related Weibos in Congestion Hotspot by Day',
                 save_filename='traffic_cgs_hotspot_per_day.png', set_percentile=97.5, color_pos='#4BB8FA',
                 color_neutral='#FAF838', color_neg='#FA6466')
hotspot_actual_day_plot(hotspot_dataframe=conges_hotspot, actual_dataframe=official_cgs_shanghai,
                        title='Congestion-related Weibos in Congestion Hotspot by Day',
                        save_filename='traffic_cgs_hotspot_with_actual_per_day.png', set_percentile=97.5,
                        color_pos='#4BB8FA', color_neutral='#FAF838', color_neg='#FA6466')
# Day plot for the congestion hotspot with different density class
hotspot_day_plot(hotspot_dataframe=cgs_hotspot_1,
                 title='Congestion-related Weibos in Congestion Hotspot by Day\n(Congestion Hotspot ID = 1)',
                 save_filename='traffic_cgs_hotspot_per_day_hotspot_1.png', set_percentile=97.5,
                 color_pos='#4BB8FA', color_neutral='#FAF838', color_neg='#FA6466')
hotspot_day_plot(hotspot_dataframe=cgs_hotspot_3,
                 title='Congestion-related Weibos in Congestion Hotspot by Day\n(Congestion Hotspot ID = 3)',
                 save_filename='traffic_cgs_hotspot_per_day_hotspot_3.png', set_percentile=97.5,
                 color_pos='#4BB8FA', color_neutral='#FAF838', color_neg='#FA6466')

