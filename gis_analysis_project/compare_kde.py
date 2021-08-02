import os
import pandas as pd
import numpy as np
from scipy.stats import halfnorm
import time
from collections import Counter
from sklearn.metrics import auc  # auc(false_positive_rates, true_positive_rates)

import geopandas as gpd
import rasterio
import shapely
shapely.speedups.disable()

from data_paths import kde_analysis, tif_save_path, fishnet_path, raster_fishnet


def compute_raster_fishnet(fishnet_filename, raster_path_filename):
    """
    Get the raster value for each fishnet grid
    :param fishnet_filename: the name of the fishnet file
    :param raster_path_filename: the name of the raster file or the path to the raster file
    :return: the geopandas shape saving the raster values
    """
    print('Load the data...')
    fishnet_shape = gpd.read_file(os.path.join(fishnet_path, fishnet_filename), encoding='utf-8')
    fishnet_shape_copy = fishnet_shape.copy()
    if os.path.exists(raster_path_filename):  # If a full path to the raster file is provided
        raster_vals = rasterio.open(raster_path_filename)
    else:  # When only the file name is provided
        raster_vals = rasterio.open(os.path.join(tif_save_path, raster_path_filename))
    assert fishnet_shape.crs.to_epsg() == raster_vals.crs.to_epsg(), "The coordinate systems do not match!"
    print('Done!')

    # Generate the centroids for the fishnet file
    fishnet_shape['centroid'] = fishnet_shape.centroid

    # Get the coordinates
    coords = [(point.x, point.y) for point in list(fishnet_shape['centroid'])]

    # Compute the raster values
    raster_vals = np.array([x for x in raster_vals.sample(coords, masked=True)])
    raster_vals[raster_vals < 0] = 0

    # Get the final fishnet file with raster values
    fishnet_shape_copy['raster_val'] = raster_vals

    return fishnet_shape_copy


def spatial_join(point_gdf, shape_area):
    """
    Find the tweets posted in one city's open space
    :param tweet_gdf: the geopandas dataframe saving the tweets
    :param shape_area: the shapefile of a studied area, such as city, open space, etc
    :return: tweets posted in open space
    """
    assert point_gdf.crs == shape_area.crs, 'The coordinate systems do not match!'
    joined_data = gpd.sjoin(left_df=point_gdf, right_df=shape_area, op='within')
    return joined_data


def compute_type_one_error(density_vals, std_error_size):
    """
    (To be updated...)
    Compute the type I error based on density values
    :param density_vals:
    :param std_error_size:
    :return: return Type I error
    """
    mean_val = np.mean(density_vals)
    std_val = np.std(density_vals)
    threshold_value = mean_val + std_error_size * std_val
    return 1 - halfnorm(loc=mean_val, scale=std_val).cdf(threshold_value)


def compute_false_alarm_miss_detection(whole_area_grid, point_feature, threshold_value, max_density_val):
    """
    Compute the miss detection rate (percentage of points not captured by the hotspot area)
    :param whole_area_grid: the whole grid shapefile (considered Shanghai fishnet)
    :param point_feature: the point shapefile
    :param threshold_value: the threshold value used to find the hotspot area
    :param max_density_val: the maximum density value used to find the hotspot
    If the threshold value is less than max_density_val, we use threshold value to find hotspot
    If the threshold value is equal or bigger than max_density_val, no spatial unit is defined as a component of hotspot
    :return: false alarm rate, miss detection rate, and a list saving [TN, FN, FP, TP]
    """
    assert 'geometry' in whole_area_grid, "The grid shapefile should contain the polygon feature"
    assert 'geometry' in point_feature, "The point shapefile should contain the point feature"

    # Count the number of city spatial units having points (accident or congestion)
    whole_regions = gpd.GeoSeries(list(whole_area_grid['geometry']))
    points = gpd.GeoSeries(list(point_feature['geometry']))
    whole_area_counter = Counter(whole_regions.apply(lambda x: points.within(x).any()))

    # Compute the false alarm, miss detection, TN, FN, FP, TP
    if threshold_value < max_density_val:
        hotspot_area = whole_area_grid.loc[whole_area_grid['raster_val'] >= threshold_value]
        not_hotspot_area = whole_area_grid.loc[whole_area_grid['raster_val'] < threshold_value]
        hotspot_regions = gpd.GeoSeries(list(hotspot_area['geometry']))
        not_hotspot_regions = gpd.GeoSeries(list(not_hotspot_area['geometry']))

        # Get the **hotspot grid units** having points
        # Get the **whole area grid units** having points
        hotspot_counter = Counter(hotspot_regions.apply(lambda x: points.within(x).any()))
        not_hotspot_counter = Counter(not_hotspot_regions.apply(lambda x: points.within(x).any()))

        print(hotspot_counter)
        print(whole_area_counter)
        miss_detection_rate = 1 - hotspot_counter[True]/whole_area_counter[True]
        false_alarm_rate = hotspot_counter[False]/whole_area_counter[False]

        # Compute for TN
        TN, FN = not_hotspot_counter[False], not_hotspot_counter[True]
        FP, TP = hotspot_counter[False], hotspot_counter[True]
        for_roc_list = [TN, FN, FP, TP]
        print('False Alarm: {}; Miss Detection: {}; TN, FN, FP, TP: {}'.format(
            false_alarm_rate, miss_detection_rate, for_roc_list))
    else:
        false_alarm_rate, miss_detection_rate = 0, 1
        for_roc_list = [whole_area_counter[False], whole_area_counter[True], 0, 0]

    return false_alarm_rate, miss_detection_rate, for_roc_list


def compute_rates_for_each_day(hotspot_grid_shape, actual_points, weibo_points):
    """
    TO BE UPDATED...
    :param hotspot_grid_shape:
    :param actual_points:
    :return:
    """
    PAI_actual = compute_pai_and_count(points=actual_points, hotspot_shape=hotspot_grid_shape)
    PAI_weibo = compute_pai_and_count(points=weibo_points, hotspot_shape=hotspot_grid_shape)
    return PAI_actual/PAI_weibo


def get_area(shape: gpd.GeoDataFrame):
    """
    Get the total area of a shape (in square kilometers)
    :param shape: the shapefile of an area
    :return: the total area of the shapefile
    """
    return round(sum(shape.area)/10**6, 4)


def main_extract_raster_for_fishnet():
    """
    Main function to compute raster for fishnet polygons
    :return: None. The created fishnet is saved to the local directory
    """
    # Create the fishnet file tif list dict
    fishnet_files = [file for file in os.listdir(fishnet_path) if ('shanghai' in file) and (file.endswith('.shp')) and
                     ('label' not in file)]
    fishnet_tif_dict = {}
    for file in fishnet_files:
        grid_cell_size = file.split('_')[2][:-4]
        select_tif_files = [file for file in os.listdir(tif_save_path) if
                            (file.endswith('.tif') and (file.split('_')[2] == grid_cell_size))]
        fishnet_tif_dict[file] = select_tif_files
    print(fishnet_tif_dict)

    # Get the raster value for each fishnet polygon
    for fishnet_file in fishnet_tif_dict:
        raster_file_list = fishnet_tif_dict[fishnet_file]
        for raster_file in raster_file_list:
            print('*' * 20)
            print('Coping with the fishnet: {}; raster file: {}'.format(fishnet_file, raster_file))
            result_fishnet = compute_raster_fishnet(fishnet_filename=fishnet_file, raster_path_filename=raster_file)
            result_fishnet.to_file(os.path.join(raster_fishnet, raster_file[:-4] + '_fishnet.shp'), encoding='utf-8')
            print('Process Done!')
            print('*' * 20)


def main_extract_raster_for_fishnet_each_day(save_path, consider_bandwidth, consider_unit_size,
                                             consider_sent=False):
    """
    Main function to compute raster for fishnet polygons
    :param: save_path: the pate used to save the created fishnet file
    :param: consider_bandwidth: the considered bandwidth length, in other words, search radius (in meter)
    :param consider_unit_size: the considered spatial unit size (in meter)
    :param consider_sent: whether consider the sentiment information or not
    :return: None. The created fishnet is saved to the local directory
    """
    # Create the fishnet file tif list dict
    fishnet_files = [file for file in os.listdir(fishnet_path) if ('shanghai' in file) and (file.endswith('.shp')) and
                     ('label' not in file) and (file[-7:-4] == str(consider_unit_size))]
    print(fishnet_files)
    fishnet_tif_dict = {}
    for file in fishnet_files:
        grid_cell_size = file[:-4].split('_')[2]
        tif_settings = os.listdir(tif_save_path)
        select_tif_files = []
        for tif_setting in tif_settings:
            check_bandwidth = tif_setting[:4]
            check_unit_size = tif_setting[-3:]
            if (str(consider_bandwidth) == check_bandwidth) and (str(consider_unit_size) == check_unit_size):
                if consider_sent:
                    select_tif_files_path = os.path.join(tif_save_path, tif_setting, 'with_sent')
                else:
                    select_tif_files_path = os.path.join(tif_save_path, tif_setting, 'without_sent')
                select_tif_files_one_setting = [file for file in os.listdir(select_tif_files_path) if
                        (file.endswith('.tif') and (file.split('_')[2] == grid_cell_size))]
                select_tif_files += select_tif_files_one_setting
        fishnet_tif_dict[file] = select_tif_files
    print(fishnet_tif_dict)
    print(len(fishnet_tif_dict))

    # Get the raster value for each fishnet polygon
    for fishnet_file in fishnet_tif_dict:
        raster_file_list = fishnet_tif_dict[fishnet_file]
        for raster_file in raster_file_list:
           print('*' * 20)
           print('Coping with the fishnet: {}; raster file: {}'.format(fishnet_file, raster_file))
           raster_setting=raster_file[4:12]
           if 'with_sent' in raster_file:
               raster_file_path = os.path.join(tif_save_path, raster_setting, 'with_sent', raster_file)
           else:
               raster_file_path = os.path.join(tif_save_path, raster_setting, 'without_sent', raster_file)
           result_fishnet = compute_raster_fishnet(fishnet_filename=fishnet_file,
                                                   raster_path_filename=raster_file_path)
           if not os.path.exists(os.path.join(save_path, raster_file[:12])):
               os.mkdir(os.path.join(save_path, raster_file[:12]))
           result_fishnet.to_file(os.path.join(save_path, raster_file[:12], raster_file[:-4] + '_fishnet.shp'),
                                  encoding='utf-8')
           print('Process Done!')
           print('*' * 20)


def compute_pai_and_count(points, hotspot_shape, shanghai_total_area=7015.05):
    """
    Compute the PAI index based on points and hotspot area shapefile
    :param points:
    :param hotspot_shape:
    :param shanghai_total_area:
    :return: PAI value based on points and hotspot area; the number of points in the hotspot
    """
    in_hotspot = spatial_join(point_gdf=points, shape_area=hotspot_shape)
    hotspot_area = get_area(hotspot_shape)
    return (in_hotspot.shape[0]/points.shape[0])/(hotspot_area/shanghai_total_area), in_hotspot.shape[0]


def build_pai_compare_dataframe(traffic_type: str, considered_bandwidth: int):
    """
    Build the traffic hotspot comparison given traffic type
    :param traffic_type: the considered traffic type
    :param considered_bandwidth: the considered bandwidth
    :return: None. The dataframe saves metrics to compare the detected hotspot
    """
    assert traffic_type in ['acc', 'cgs'], "The traffic type should be either acc or cgs"
    assert considered_bandwidth in [1000, 2000, 3000], "The bandwidth should be either 1000, 2000, or 3000"

    # Specify the paths to load the data
    shapefile_path = os.path.join(kde_analysis, 'shapefile')

    if traffic_type == 'acc':
        shapefiles_with_sent = [file for file in os.listdir(raster_fishnet) if
                                file.startswith('acc') and file.endswith('with_sent_fishnet.shp') and
                                (str(considered_bandwidth) in file)]
        shapefiles_without_sent = [file for file in os.listdir(raster_fishnet) if
                                   file.startswith('acc') and file.endswith('without_sent_fishnet.shp') and
                                   (str(considered_bandwidth) in file)]
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_accident.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_accident_shanghai.shp'), encoding='utf-8')
    else:
        shapefiles_with_sent = [file for file in os.listdir(raster_fishnet) if
                                file.startswith('cgs') and file.endswith('with_sent_fishnet.shp') and
                                (str(considered_bandwidth) in file)]
        shapefiles_without_sent = [file for file in os.listdir(raster_fishnet) if
                                   file.startswith('cgs') and file.endswith('without_sent_fishnet.shp') and
                                   (str(considered_bandwidth) in file)]
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_congestion.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_congestion_shanghai.shp'),
                                       encoding='utf-8')

    print('Considered shapefiles with sentiment information: {}'.format(shapefiles_with_sent))
    print('Considered shapefiles without sentiment information: {}'.format(shapefiles_without_sent))

    setting_list, traffic_type_list, bandwidth_list, unit_size_list = [], [], [], []
    consider_sent_list, PAI_weibo_list, PAI_actual_list, hotspot_area_list = [], [], [], []
    weibo_in_hotspot_count_list, actual_in_hotspot_count_list, threshold_value_list = [], [], []
    false_alarm_rates_weibo, miss_detection_rates_weibo = [], []
    false_alarm_rates_actual, miss_detection_rates_actual = [], []
    TN_weibo_list, FN_weibo_list, FP_weibo_list, TP_weibo_list = [], [], [], []
    TN_actual_list, FN_actual_list, FP_actual_list, TP_actual_list = [], [], [], []

    print('*' * 20)
    print('Coping with the hotspot with sentiment')
    for shapefile in shapefiles_with_sent:
        print('Coping with the file: {}'.format(shapefile))
        consider_fishnet = gpd.read_file(os.path.join(raster_fishnet, shapefile), encoding='utf-8')
        raster_vals = list(consider_fishnet['raster_val'])
        threshold_val = np.mean(raster_vals) + 3 * np.std(raster_vals)
        max_density_value = np.max(raster_vals)
        print('Treshold: {}'.format(threshold_val))
        setting_list.append(shapefile[:-4])
        traffic_type_list.append(traffic_type)
        consider_sent_list.append('with_sentiment')
        bandwidth = shapefile[:-4].split('_')[1]
        unit_size = shapefile[:-4].split('_')[2]
        bandwidth_list.append(bandwidth)
        unit_size_list.append(unit_size)
        threshold_value_list.append(threshold_val)
        hotspot_area = consider_fishnet.loc[consider_fishnet['raster_val'] >= threshold_val]
        consider_hotspot_area = get_area(hotspot_area)
        hotspot_area_list.append(consider_hotspot_area)
        PAI_weibo, weibo_count_in_hotspot = compute_pai_and_count(points=weibo_point,
                                                                  hotspot_shape=hotspot_area)
        PAI_actual, actual_count_in_hotspot = compute_pai_and_count(points=official_point,
                                                                    hotspot_shape=hotspot_area)
        weibo_in_hotspot_count_list.append(weibo_count_in_hotspot)
        actual_in_hotspot_count_list.append(actual_count_in_hotspot)
        PAI_weibo_list.append(PAI_weibo)
        PAI_actual_list.append(PAI_actual)
        false_rate_weibo, miss_rate_weibo, roc_list_weibo = compute_false_alarm_miss_detection(
            whole_area_grid=consider_fishnet, point_feature=weibo_point, threshold_value=threshold_val,
            max_density_val=max_density_value)
        false_alarm_rates_weibo.append(false_rate_weibo)
        miss_detection_rates_weibo.append(miss_rate_weibo)
        TN_weibo_list.append(roc_list_weibo[0])  # TN
        FN_weibo_list.append(roc_list_weibo[1])  # FN
        FP_weibo_list.append(roc_list_weibo[2])  # FP
        TP_weibo_list.append(roc_list_weibo[3])  # TP
        false_rate_actual, miss_rate_actual, roc_list_actual = compute_false_alarm_miss_detection(
            whole_area_grid=consider_fishnet,
            point_feature=official_point,
            threshold_value=threshold_val,
            max_density_val=max_density_value)
        false_alarm_rates_actual.append(false_rate_actual)
        miss_detection_rates_actual.append(miss_rate_actual)
        TN_actual_list.append(roc_list_actual[0])  # TN
        FN_actual_list.append(roc_list_actual[1])  # FN
        FP_actual_list.append(roc_list_actual[2])  # FP
        TP_actual_list.append(roc_list_actual[3])  # TP

    print('*' * 20)
    print('Coping with the hotspot without sentiment')
    for shapefile in shapefiles_without_sent:
        print('Coping with the file: {}'.format(shapefile))
        consider_fishnet = gpd.read_file(os.path.join(raster_fishnet, shapefile), encoding='utf-8')
        raster_vals = list(consider_fishnet['raster_val'])
        threshold_val = np.mean(raster_vals) + 3 * np.std(raster_vals)
        max_density_value = np.max(raster_vals)
        print('Treshold: {}'.format(threshold_val))
        setting_list.append(shapefile[:-4])
        traffic_type_list.append(traffic_type)
        consider_sent_list.append('without_sentiment')
        bandwidth = shapefile[:-4].split('_')[1]
        unit_size = shapefile[:-4].split('_')[2]
        bandwidth_list.append(bandwidth)
        unit_size_list.append(unit_size)
        threshold_value_list.append(threshold_val)
        hotspot_area = consider_fishnet.loc[consider_fishnet['raster_val'] >= threshold_val]
        consider_hotspot_area = get_area(hotspot_area)
        hotspot_area_list.append(consider_hotspot_area)
        PAI_weibo, weibo_count_in_hotspot = compute_pai_and_count(points=weibo_point,
                                                                  hotspot_shape=hotspot_area)
        PAI_actual, actual_count_in_hotspot = compute_pai_and_count(points=official_point,
                                                                    hotspot_shape=hotspot_area)
        weibo_in_hotspot_count_list.append(weibo_count_in_hotspot)
        actual_in_hotspot_count_list.append(actual_count_in_hotspot)
        PAI_weibo_list.append(PAI_weibo)
        PAI_actual_list.append(PAI_actual)
        false_rate_weibo, miss_rate_weibo, roc_list_weibo = compute_false_alarm_miss_detection(
            whole_area_grid=consider_fishnet, point_feature=weibo_point, threshold_value=threshold_val,
            max_density_val=max_density_value)
        false_alarm_rates_weibo.append(false_rate_weibo)
        miss_detection_rates_weibo.append(miss_rate_weibo)
        TN_weibo_list.append(roc_list_weibo[0])  # TN
        FN_weibo_list.append(roc_list_weibo[1])  # FN
        FP_weibo_list.append(roc_list_weibo[2])  # FP
        TP_weibo_list.append(roc_list_weibo[3])  # TP
        false_rate_actual, miss_rate_actual, roc_list_actual = compute_false_alarm_miss_detection(
            whole_area_grid=consider_fishnet,
            point_feature=official_point,
            threshold_value=threshold_val,
            max_density_val=max_density_value)
        false_alarm_rates_actual.append(false_rate_actual)
        miss_detection_rates_actual.append(miss_rate_actual)
        TN_actual_list.append(roc_list_actual[0])  # TN
        FN_actual_list.append(roc_list_actual[1])  # FN
        FP_actual_list.append(roc_list_actual[2])  # FP
        TP_actual_list.append(roc_list_actual[3])  # TP

    # Incrementally save the hotspot validation result to the dataframe...
    print('Save the dataframe for hotspot module validation...')
    pai_dataframe = pd.DataFrame()
    pai_dataframe['setting'] = setting_list
    pai_dataframe['traffic_type'] = traffic_type_list
    pai_dataframe['consider_sent'] = consider_sent_list
    pai_dataframe['weibo_in_hotspot'] = weibo_in_hotspot_count_list
    pai_dataframe['actual_in_hotspot'] = actual_in_hotspot_count_list
    pai_dataframe['hotspot_area'] = hotspot_area_list
    pai_dataframe['bandwidth'] = bandwidth_list
    pai_dataframe['unit_size'] = unit_size_list
    pai_dataframe['threshold_val'] = threshold_value_list
    pai_dataframe['PAI_weibo'] = PAI_weibo_list
    pai_dataframe['PAI_actual'] = PAI_actual_list
    pai_dataframe['PAI_ratio'] = pai_dataframe['PAI_actual'] / pai_dataframe['PAI_weibo']
    pai_dataframe['false_alarm_weibo'] = false_alarm_rates_weibo
    pai_dataframe['miss_detection_weibo'] = miss_detection_rates_weibo
    pai_dataframe['false_alarm_actual'] = false_alarm_rates_actual
    pai_dataframe['miss_detection_actual'] = miss_detection_rates_actual
    pai_dataframe['TN_weibo'] = TN_weibo_list
    pai_dataframe['FN_weibo'] = FN_weibo_list
    pai_dataframe['FP_weibo'] = FP_weibo_list
    pai_dataframe['TP_weibo'] = TP_weibo_list
    pai_dataframe['TN_actual'] = TN_actual_list
    pai_dataframe['FN_actual'] = FN_actual_list
    pai_dataframe['FP_actual'] = FP_actual_list
    pai_dataframe['TP_actual'] = TP_actual_list

    # Compute TPR and FPR for Weibos and actual records
    pai_dataframe['TPR_weibo'] = pai_dataframe['TP_weibo'] / (
            pai_dataframe['TP_weibo'] + pai_dataframe['FN_weibo'])
    pai_dataframe['FPR_weibo'] = pai_dataframe['FP_weibo'] / (
            pai_dataframe['FP_weibo'] + pai_dataframe['TN_weibo'])
    pai_dataframe['TPR_actual'] = pai_dataframe['TP_actual'] / (
            pai_dataframe['TP_actual'] + pai_dataframe['FN_actual'])
    pai_dataframe['FPR_actual'] = pai_dataframe['FP_actual'] / (
            pai_dataframe['FP_actual'] + pai_dataframe['TN_actual'])
    pai_dataframe.to_csv(os.path.join(kde_analysis, 'kde_compare',
                                      '{}_kde_compare_given_threshold.csv'.format(traffic_type)),
                         encoding='utf-8')


def build_pai_compare_dataframe_multiple_thresholds(traffic_type: str, threshold_num: int = 50) -> pd.DataFrame:
    """
    Build the dataframe for hotspot comparison
    :param traffic_type: the type of considered traffic, either 'acc' or 'cgs'
    :param threshold_num: the number of considered threshold values for hotspot comparison
    :return: a pandas dataframe saving the PAI, number of traffic-relevant Weibos, etc. for comparison
    """
    assert traffic_type in ['acc', 'cgs'], "The traffic type should be either acc or cgs"

    # Specify the paths to load the data
    shapefile_path = os.path.join(kde_analysis, 'shapefile')

    if traffic_type == 'acc':
        shapefiles_with_sent = [file for file in os.listdir(raster_fishnet) if
                                file.startswith('acc') and file.endswith('with_sent_fishnet.shp')]
        shapefiles_without_sent = [file for file in os.listdir(raster_fishnet) if
                                   file.startswith('acc') and file.endswith('without_sent_fishnet.shp')]
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_accident.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_accident_shanghai.shp'), encoding='utf-8')
    else:
        shapefiles_with_sent = [file for file in os.listdir(raster_fishnet) if
                                file.startswith('cgs') and file.endswith('with_sent_fishnet.shp')]
        shapefiles_without_sent = [file for file in os.listdir(raster_fishnet) if
                                   file.startswith('cgs') and file.endswith('without_sent_fishnet.shp')]
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_congestion.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_congestion_shanghai.shp'),
                                       encoding='utf-8')

    print('Considered shapefiles with sentiment information: {}'.format(shapefiles_with_sent))
    print('Considered shapefiles without sentiment information: {}'.format(shapefiles_without_sent))

    setting_list, traffic_type_list, bandwidth_list, unit_size_list = [], [], [], []
    consider_sent_list, PAI_weibo_list, PAI_actual_list, hotspot_area_list = [], [], [], []
    weibo_in_hotspot_count_list, actual_in_hotspot_count_list, threshold_value_list = [], [], []
    false_alarm_rates_weibo, miss_detection_rates_weibo = [], []
    false_alarm_rates_actual, miss_detection_rates_actual = [], []
    TN_weibo_list, FN_weibo_list, FP_weibo_list, TP_weibo_list = [], [], [], []
    TN_actual_list, FN_actual_list, FP_actual_list, TP_actual_list = [], [], [], []

    print('*' * 20)
    print('Coping with the hotspot with sentiment')
    shapefile_counter = 0
    for shapefile in shapefiles_with_sent:
        print('Coping with the file: {}'.format(shapefile))
        consider_fishnet = gpd.read_file(os.path.join(raster_fishnet, shapefile), encoding='utf-8')
        raster_vals = list(consider_fishnet['raster_val'])
        # density_val_mean = np.mean(raster_vals)
        # density_val_std = np.std(raster_vals)
        # chose_threshold = density_val_mean + 3 * density_val_std  # mean + 3 * std
        density_threshold_low = 0
        density_threshold_up = max(raster_vals)
        threshold_choices = np.linspace(density_threshold_low, density_threshold_up, threshold_num)
        # threshold_choices = np.append(threshold_choices, chose_threshold)
        for threshold_val in threshold_choices:
            print('Treshold: {}'.format(threshold_val))
            setting_list.append(shapefile[:-4])
            traffic_type_list.append(traffic_type)
            consider_sent_list.append('with_sentiment')
            bandwidth = shapefile[:-4].split('_')[1]
            unit_size = shapefile[:-4].split('_')[2]
            bandwidth_list.append(bandwidth)
            unit_size_list.append(unit_size)
            threshold_value_list.append(threshold_val)
            if threshold_val < density_threshold_up:
                hotspot_area = consider_fishnet.loc[consider_fishnet['raster_val'] >= threshold_val]
                consider_hotspot_area = get_area(hotspot_area)
                hotspot_area_list.append(consider_hotspot_area)
                PAI_weibo, weibo_count_in_hotspot = compute_pai_and_count(points=weibo_point,
                                                                          hotspot_shape=hotspot_area)
                PAI_actual, actual_count_in_hotspot = compute_pai_and_count(points=official_point,
                                                                            hotspot_shape=hotspot_area)
            else:
                hotspot_area_list.append(0)
                weibo_count_in_hotspot = 0
                actual_count_in_hotspot = 0
                PAI_weibo, PAI_actual = 0, 0
            weibo_in_hotspot_count_list.append(weibo_count_in_hotspot)
            actual_in_hotspot_count_list.append(actual_count_in_hotspot)
            PAI_weibo_list.append(PAI_weibo)
            PAI_actual_list.append(PAI_actual)
            false_rate_weibo, miss_rate_weibo, roc_list_weibo = compute_false_alarm_miss_detection(
                whole_area_grid=consider_fishnet, point_feature=weibo_point, threshold_value=threshold_val,
                max_density_val=density_threshold_up)
            false_alarm_rates_weibo.append(false_rate_weibo)
            miss_detection_rates_weibo.append(miss_rate_weibo)
            TN_weibo_list.append(roc_list_weibo[0])  # TN
            FN_weibo_list.append(roc_list_weibo[1])  # FN
            FP_weibo_list.append(roc_list_weibo[2])  # FP
            TP_weibo_list.append(roc_list_weibo[3])  # TP
            false_rate_actual, miss_rate_actual, roc_list_actual = compute_false_alarm_miss_detection(
                whole_area_grid=consider_fishnet,
                point_feature=official_point,
                threshold_value=threshold_val,
                max_density_val=density_threshold_up)
            false_alarm_rates_actual.append(false_rate_actual)
            miss_detection_rates_actual.append(miss_rate_actual)
            TN_actual_list.append(roc_list_actual[0])  # TN
            FN_actual_list.append(roc_list_actual[1])  # FN
            FP_actual_list.append(roc_list_actual[2])  # FP
            TP_actual_list.append(roc_list_actual[3])  # TP

        # Incrementally save the hotspot validation result to the dataframe...
        print('Save the dataframe for hotspot module validation...')
        pai_dataframe = pd.DataFrame()
        pai_dataframe['setting'] = setting_list
        pai_dataframe['traffic_type'] = traffic_type_list
        pai_dataframe['consider_sent'] = consider_sent_list
        pai_dataframe['weibo_in_hotspot'] = weibo_in_hotspot_count_list
        pai_dataframe['actual_in_hotspot'] = actual_in_hotspot_count_list
        pai_dataframe['hotspot_area'] = hotspot_area_list
        pai_dataframe['bandwidth'] = bandwidth_list
        pai_dataframe['unit_size'] = unit_size_list
        pai_dataframe['threshold_val'] = threshold_value_list
        pai_dataframe['PAI_weibo'] = PAI_weibo_list
        pai_dataframe['PAI_actual'] = PAI_actual_list
        pai_dataframe['PAI_ratio'] = pai_dataframe['PAI_actual'] / pai_dataframe['PAI_weibo']
        pai_dataframe['false_alarm_weibo'] = false_alarm_rates_weibo
        pai_dataframe['miss_detection_weibo'] = miss_detection_rates_weibo
        pai_dataframe['false_alarm_actual'] = false_alarm_rates_actual
        pai_dataframe['miss_detection_actual'] = miss_detection_rates_actual
        pai_dataframe['TN_weibo'] = TN_weibo_list
        pai_dataframe['FN_weibo'] = FN_weibo_list
        pai_dataframe['FP_weibo'] = FP_weibo_list
        pai_dataframe['TP_weibo'] = TP_weibo_list
        pai_dataframe['TN_actual'] = TN_actual_list
        pai_dataframe['FN_actual'] = FN_actual_list
        pai_dataframe['FP_actual'] = FP_actual_list
        pai_dataframe['TP_actual'] = TP_actual_list

        # Compute TPR and FPR for Weibos and actual records
        try:
            pai_dataframe['TPR_weibo'] = pai_dataframe['TP_weibo'] / (
                        pai_dataframe['TP_weibo'] + pai_dataframe['FN_weibo'])
            pai_dataframe['FPR_weibo'] = pai_dataframe['FP_weibo'] / (
                        pai_dataframe['FP_weibo'] + pai_dataframe['TN_weibo'])
            pai_dataframe['TPR_actual'] = pai_dataframe['TP_actual'] / (
                        pai_dataframe['TP_actual'] + pai_dataframe['FN_actual'])
            pai_dataframe['FPR_actual'] = pai_dataframe['FP_actual'] / (
                        pai_dataframe['FP_actual'] + pai_dataframe['TN_actual'])
            pai_dataframe.to_csv(os.path.join(kde_analysis, 'kde_compare',
                                              '{}_kde_compare_{}.csv'.format(traffic_type,
                                                                                  shapefile_counter)),
                                 encoding='utf-8')
            shapefile_counter += 1
        except:
            print('Some error occurs! Save the data first...')
            pai_dataframe.to_csv(os.path.join(kde_analysis, 'kde_compare',
                                              '{}_kde_compare_{}.csv'.format(traffic_type,
                                                                                  shapefile_counter)),
                                 encoding='utf-8')
            shapefile_counter += 1
        print('Done!')
    print('*' * 20 + '\n')

    print('*' * 20)
    print('Coping with the hotspot without sentiment')
    for shapefile in shapefiles_without_sent:
        print('Coping with the file: {}'.format(shapefile))
        consider_fishnet = gpd.read_file(os.path.join(raster_fishnet, shapefile), encoding='utf-8')
        raster_vals = list(consider_fishnet['raster_val'])
        # density_val_mean = np.mean(raster_vals)
        # density_val_std = np.std(raster_vals)
        # chose_threshold = density_val_mean + 3 * density_val_std  # mean + 3 * std
        density_threshold_low = 0
        density_threshold_up = max(raster_vals)
        threshold_choices = np.linspace(density_threshold_low, density_threshold_up, threshold_num)
        # threshold_choices = np.append(threshold_choices, chose_threshold)
        for threshold_val in threshold_choices:
            setting_list.append(shapefile[:-4])
            traffic_type_list.append(traffic_type)
            consider_sent_list.append('without_sentiment')
            bandwidth = shapefile[:-4].split('_')[1]
            unit_size = shapefile[:-4].split('_')[2]
            bandwidth_list.append(bandwidth)
            unit_size_list.append(unit_size)
            threshold_value_list.append(threshold_val)
            if threshold_val < density_threshold_up:
                hotspot_area = consider_fishnet.loc[consider_fishnet['raster_val'] >= threshold_val]
                consider_hotspot_area = get_area(hotspot_area)
                hotspot_area_list.append(consider_hotspot_area)
                PAI_weibo, weibo_count_in_hotspot = compute_pai_and_count(points=weibo_point,
                                                                          hotspot_shape=hotspot_area)
                PAI_actual, actual_count_in_hotspot = compute_pai_and_count(points=official_point,
                                                                            hotspot_shape=hotspot_area)
            else:
                hotspot_area_list.append(0)
                weibo_count_in_hotspot = 0
                actual_count_in_hotspot = 0
                PAI_weibo, PAI_actual = 0, 0
            weibo_in_hotspot_count_list.append(weibo_count_in_hotspot)
            actual_in_hotspot_count_list.append(actual_count_in_hotspot)
            PAI_weibo_list.append(PAI_weibo)
            PAI_actual_list.append(PAI_actual)
            false_rate_weibo, miss_rate_weibo, roc_list_weibo = compute_false_alarm_miss_detection(
                whole_area_grid=consider_fishnet, point_feature=weibo_point, threshold_value=threshold_val,
                max_density_val=density_threshold_up)
            false_alarm_rates_weibo.append(false_rate_weibo)
            miss_detection_rates_weibo.append(miss_rate_weibo)
            TN_weibo_list.append(roc_list_weibo[0])  # TN
            FN_weibo_list.append(roc_list_weibo[1])  # FN
            FP_weibo_list.append(roc_list_weibo[2])  # FP
            TP_weibo_list.append(roc_list_weibo[3])  # TP
            false_rate_actual, miss_rate_actual, roc_list_actual = compute_false_alarm_miss_detection(
                whole_area_grid=consider_fishnet,
                point_feature=official_point,
                threshold_value=threshold_val,
                max_density_val=density_threshold_up)
            false_alarm_rates_actual.append(false_rate_actual)
            miss_detection_rates_actual.append(miss_rate_actual)
            TN_actual_list.append(roc_list_actual[0])  # TN
            FN_actual_list.append(roc_list_actual[1])  # FN
            FP_actual_list.append(roc_list_actual[2])  # FP
            TP_actual_list.append(roc_list_actual[3])  # TP

        # Incrementally save the hotspot validation result to the dataframe...
        print('Save the dataframe for hotspot module validation...')
        pai_dataframe = pd.DataFrame()
        pai_dataframe['setting'] = setting_list
        pai_dataframe['traffic_type'] = traffic_type_list
        pai_dataframe['consider_sent'] = consider_sent_list
        pai_dataframe['weibo_in_hotspot'] = weibo_in_hotspot_count_list
        pai_dataframe['actual_in_hotspot'] = actual_in_hotspot_count_list
        pai_dataframe['hotspot_area'] = hotspot_area_list
        pai_dataframe['bandwidth'] = bandwidth_list
        pai_dataframe['unit_size'] = unit_size_list
        pai_dataframe['threshold_val'] = threshold_value_list
        pai_dataframe['PAI_weibo'] = PAI_weibo_list
        pai_dataframe['PAI_actual'] = PAI_actual_list
        pai_dataframe['PAI_ratio'] = pai_dataframe['PAI_actual'] / pai_dataframe['PAI_weibo']
        pai_dataframe['false_alarm_weibo'] = false_alarm_rates_weibo
        pai_dataframe['miss_detection_weibo'] = miss_detection_rates_weibo
        pai_dataframe['false_alarm_actual'] = false_alarm_rates_actual
        pai_dataframe['miss_detection_actual'] = miss_detection_rates_actual
        pai_dataframe['TN_weibo'] = TN_weibo_list
        pai_dataframe['FN_weibo'] = FN_weibo_list
        pai_dataframe['FP_weibo'] = FP_weibo_list
        pai_dataframe['TP_weibo'] = TP_weibo_list
        pai_dataframe['TN_actual'] = TN_actual_list
        pai_dataframe['FN_actual'] = FN_actual_list
        pai_dataframe['FP_actual'] = FP_actual_list
        pai_dataframe['TP_actual'] = TP_actual_list

        # Compute TPR and FPR for Weibos and actual records
        try:
            pai_dataframe['TPR_weibo'] = pai_dataframe['TP_weibo'] / (
                    pai_dataframe['TP_weibo'] + pai_dataframe['FN_weibo'])
            pai_dataframe['FPR_weibo'] = pai_dataframe['FP_weibo'] / (
                    pai_dataframe['FP_weibo'] + pai_dataframe['TN_weibo'])
            pai_dataframe['TPR_actual'] = pai_dataframe['TP_actual'] / (
                    pai_dataframe['TP_actual'] + pai_dataframe['FN_actual'])
            pai_dataframe['FPR_actual'] = pai_dataframe['FP_actual'] / (
                    pai_dataframe['FP_actual'] + pai_dataframe['TN_actual'])
            pai_dataframe.to_csv(os.path.join(kde_analysis, 'kde_compare',
                                              '{}_kde_compare_{}.csv'.format(traffic_type,
                                                                                  shapefile_counter)),
                                 encoding='utf-8')
            shapefile_counter += 1
        except:
            print('Some error occurs! Save the data first...')
            pai_dataframe.to_csv(os.path.join(kde_analysis, 'kde_compare',
                                              '{}_kde_compare_{}.csv'.format(traffic_type,
                                                                                  shapefile_counter)),
                                 encoding='utf-8')
            shapefile_counter += 1
        print('Done!')
    print('*' * 20)


def build_pai_dataframe_for_each_day(traffic_type, consider_sent=False):
    """
    Build the pai validation dataframe for each day
    :param traffic_type: the considered traffic type, either 'acc' or 'cgs'
    :param consider_sent: whether consider the sentiment information or not
    :return: None. The daily validation dataframe is saved to local
    """
    assert traffic_type in ['acc', 'cgs'], "The traffic type should be either acc or cgs"

    # Specify the paths to load the data
    shapefile_path = os.path.join(kde_analysis, 'shapefile')

    # Load the point and fishnet shapefiles
    if traffic_type == 'acc':
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_accident.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_accident_shanghai.shp'), encoding='utf-8')
        considered_fishnet_path = os.path.join(kde_analysis, 'raster_polygons', 'acc_2000_260')
    else:
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_congestion.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_congestion_shanghai.shp'),
                                       encoding='utf-8')
        considered_fishnet_path = os.path.join(kde_analysis, 'raster_polygons', 'cgs_2000_260')
    if consider_sent:
        considered_fishnet_shapefiles = [file for file in os.listdir(considered_fishnet_path) if
                                         (file.endswith('.shp')) and ('with_sent' in file)]
    else:
        considered_fishnet_shapefiles = [file for file in os.listdir(considered_fishnet_path) if
                                         (file.endswith('.shp')) and ('without_sent' in file)]
    print(considered_fishnet_shapefiles)

    # Create the list saving the metrics for hotspot daily validation
    setting_list, traffic_type_list = [], []
    bandwidth_list, unit_size_list = [], []
    month_list, day_list = [], []
    PAI_weibo_list, PAI_actual_list, hotspot_area_list =  [], [], []
    weibo_in_hotspot_count_list, actual_in_hotspot_count_list, threshold_value_list = [], [], []
    false_alarm_rates_weibo, miss_detection_rates_weibo = [], []
    false_alarm_rates_actual, miss_detection_rates_actual = [], []
    TN_weibo_list, FN_weibo_list, FP_weibo_list, TP_weibo_list = [], [], [], []
    TN_actual_list, FN_actual_list, FP_actual_list, TP_actual_list = [], [], [], []

    # Compute the metrics for hotspot daily validation
    print('*' * 20)
    for shapefile in considered_fishnet_shapefiles:
        print('Coping with the file: {}'.format(shapefile))
        consider_fishnet = gpd.read_file(os.path.join(considered_fishnet_path, shapefile), encoding='utf-8')
        raster_vals = list(consider_fishnet['raster_val'])
        threshold_val = np.mean(raster_vals) + 3 * np.std(raster_vals)  # still setting the threshold as mean + 3*std
        density_threshold_up = np.max(raster_vals)
        print('Treshold: {}'.format(threshold_val))
        setting_list.append(shapefile[:-4])
        traffic_type_list.append(traffic_type)
        bandwidth = shapefile[:-4].split('_')[1]
        unit_size = shapefile[:-4].split('_')[2]
        month = int(shapefile[:-4].split('_')[3])
        day = int(shapefile[:-4].split('_')[4])
        bandwidth_list.append(bandwidth)
        unit_size_list.append(unit_size)
        month_list.append(month)
        day_list.append(day)

        threshold_value_list.append(threshold_val)
        select_weibo_point = weibo_point.loc[(weibo_point['month'] == month) & (weibo_point['day'] == day)]
        select_actual_point = official_point.loc[(official_point['month'] == month) & (official_point['day'] == day)]
        print('Number of {} relevant Weibos on {}-{}: {}'.format(traffic_type, month, day, select_weibo_point.shape[0]))
        print('Number of {} relevant records on {}-{}: {}'.format(traffic_type, month, day,
                                                                  select_actual_point.shape[0]))
        hotspot_area = consider_fishnet.loc[consider_fishnet['raster_val'] >= threshold_val]
        not_hotspot_area = consider_fishnet.loc[consider_fishnet['raster_val'] < threshold_val]
        consider_hotspot_area = get_area(hotspot_area)
        hotspot_area_list.append(consider_hotspot_area)
        print('Computing the PAI....')
        PAI_weibo, weibo_count_in_hotspot = compute_pai_and_count(points=select_weibo_point,
                                                                  hotspot_shape=hotspot_area)
        if select_actual_point.shape[0] > 1:
            PAI_actual, actual_count_in_hotspot = compute_pai_and_count(points=select_actual_point,
                                                                        hotspot_shape=hotspot_area)
        else:
            actual_count_in_hotspot = 0
            PAI_actual = 0
        weibo_in_hotspot_count_list.append(weibo_count_in_hotspot)
        actual_in_hotspot_count_list.append(actual_count_in_hotspot)
        PAI_weibo_list.append(PAI_weibo)
        PAI_actual_list.append(PAI_actual)
        print('Done!')
        print('Computing the false alarm, miss detection, tpr, and fpr')
        false_rate_weibo, miss_rate_weibo, roc_list_weibo = compute_false_alarm_miss_detection(
            whole_area_grid=consider_fishnet, point_feature=select_weibo_point, threshold_value=threshold_val,
            max_density_val=density_threshold_up)
        false_alarm_rates_weibo.append(false_rate_weibo)
        miss_detection_rates_weibo.append(miss_rate_weibo)
        TN_weibo_list.append(roc_list_weibo[0])  # TN
        FN_weibo_list.append(roc_list_weibo[1])  # FN
        FP_weibo_list.append(roc_list_weibo[2])  # FP
        TP_weibo_list.append(roc_list_weibo[3])  # TP
        if select_actual_point.shape[0] > 1:
            false_rate_actual, miss_rate_actual, roc_list_actual = compute_false_alarm_miss_detection(
                whole_area_grid=consider_fishnet,
                point_feature=select_actual_point,
                threshold_value=threshold_val,
                max_density_val=density_threshold_up)
            false_alarm_rates_actual.append(false_rate_actual)
            miss_detection_rates_actual.append(miss_rate_actual)
            TN_actual_list.append(roc_list_actual[0])  # TN
            FN_actual_list.append(roc_list_actual[1])  # FN
            FP_actual_list.append(roc_list_actual[2])  # FP
            TP_actual_list.append(roc_list_actual[3])  # TP
        else:
            false_rate_actual = hotspot_area.shape[0]/consider_fishnet.shape[0]
            miss_rate_actual = 'no actual'
            false_alarm_rates_actual.append(false_rate_actual)
            miss_detection_rates_actual.append(miss_rate_actual)
            TN_actual_list.append(not_hotspot_area.shape[0])  # TN
            FN_actual_list.append(0)  # FN
            FP_actual_list.append(hotspot_area.shape[0])  # FP
            TP_actual_list.append(0)  # TP
        print('Done')


    # Incrementally save the hotspot validation result to the dataframe...
    print('Save the dataframe for hotspot module validation...')
    pai_dataframe = pd.DataFrame()
    pai_dataframe['setting'] = setting_list
    pai_dataframe['month'] = month_list
    pai_dataframe['day'] = day_list
    pai_dataframe['traffic_type'] = traffic_type_list
    pai_dataframe['weibo_in_hotspot'] = weibo_in_hotspot_count_list
    pai_dataframe['actual_in_hotspot'] = actual_in_hotspot_count_list
    pai_dataframe['hotspot_area'] = hotspot_area_list
    pai_dataframe['bandwidth'] = bandwidth_list
    pai_dataframe['unit_size'] = unit_size_list
    pai_dataframe['threshold_val'] = threshold_value_list
    pai_dataframe['PAI_weibo'] = PAI_weibo_list
    pai_dataframe['PAI_actual'] = PAI_actual_list
    pai_dataframe['PAI_ratio'] = pai_dataframe['PAI_actual'] / pai_dataframe['PAI_weibo']
    pai_dataframe['false_alarm_weibo'] = false_alarm_rates_weibo
    pai_dataframe['miss_detection_weibo'] = miss_detection_rates_weibo
    pai_dataframe['false_alarm_actual'] = false_alarm_rates_actual
    pai_dataframe['miss_detection_actual'] = miss_detection_rates_actual
    pai_dataframe['TN_weibo'] = TN_weibo_list
    pai_dataframe['FN_weibo'] = FN_weibo_list
    pai_dataframe['FP_weibo'] = FP_weibo_list
    pai_dataframe['TP_weibo'] = TP_weibo_list
    pai_dataframe['TN_actual'] = TN_actual_list
    pai_dataframe['FN_actual'] = FN_actual_list
    pai_dataframe['FP_actual'] = FP_actual_list
    pai_dataframe['TP_actual'] = TP_actual_list

    # Compute TPR and FPR for Weibos and actual records
    pai_dataframe['TPR_weibo'] = pai_dataframe['TP_weibo'] / (
            pai_dataframe['TP_weibo'] + pai_dataframe['FN_weibo'])
    pai_dataframe['FPR_weibo'] = pai_dataframe['FP_weibo'] / (
            pai_dataframe['FP_weibo'] + pai_dataframe['TN_weibo'])
    pai_dataframe['TPR_actual'] = pai_dataframe['TP_actual'] / (
            pai_dataframe['TP_actual'] + pai_dataframe['FN_actual'])
    pai_dataframe['FPR_actual'] = pai_dataframe['FP_actual'] / (
            pai_dataframe['FP_actual'] + pai_dataframe['TN_actual'])
    pai_dataframe.to_csv(os.path.join(kde_analysis, 'kde_compare',
                                      '{}_daily_validate_three_std.csv'.format(traffic_type)),
                         encoding='utf-8')


if __name__ == '__main__':

    # # Get the fishnet having raster values
    # main_extract_raster_for_fishnet()

    # Compare the kde hotspots based on different threshold
    starting_time = time.time()
    # build_pai_compare_dataframe_multiple_thresholds(traffic_type='acc', threshold_num=70)
    # build_pai_compare_dataframe_multiple_thresholds(traffic_type='cgs', threshold_num=70)
    build_pai_compare_dataframe(traffic_type='acc', considered_bandwidth=2000)
    build_pai_compare_dataframe(traffic_type='cgs', considered_bandwidth=2000)
    ending_time = time.time()
    print('Total time: {}'.format(round((ending_time - starting_time)/3600, 2)))


if __name__ == '__main__':

    # # Get the fishnet having raster values
    # main_extract_raster_for_fishnet()

    # Compare the kde hotspots based on different threshold
    starting_time = time.time()
    build_pai_compare_dataframe_multiple_thresholds(traffic_type='acc', threshold_num=70)
    build_pai_compare_dataframe_multiple_thresholds(traffic_type='cgs', threshold_num=70)
    build_pai_compare_dataframe(traffic_type='acc', considered_bandwidth=2000)
    build_pai_compare_dataframe(traffic_type='cgs', considered_bandwidth=2000)
    ending_time = time.time()
    print('Total time: {}'.format(round((ending_time - starting_time)/3600, 2)))