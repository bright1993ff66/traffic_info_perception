#!D:\Projects\Traffic_info_perception\Scripts\python.exe
# -*- coding: utf-8 -*-
# basics
import os
import pandas as pd
import numpy as np
import time
from collections import Counter
from sklearn.metrics import auc  # auc(false_positive_rates, true_positive_rates)

# for spatial analysis
import geopandas as gpd
import rasterio
import shapely

# load some paths
from data_paths import kde_analysis, tif_save_path, fishnet_path, raster_fishnet

shapely.speedups.disable()


def find_related_official_fishnet(weibo_shapefile_name, path_to_fishnet=raster_fishnet):
    """
    Find the corresponding official fishnet files with KDE values to compute hotspot precision
    :param weibo_shapefile_name: the shapefile for Weibo fishnet
    :param path_to_fishnet: the path saving the fishnet
    :return: the name of the corresponding official fishnet file
    """
    if 'acc' in weibo_shapefile_name:
        traffic_type = 'acc'
    else:
        traffic_type = 'cgs'

    # Get the considered unit size and bandwidth
    unit_size = weibo_shapefile_name[:-4].split('_')[2]
    considered_bandwidth = weibo_shapefile_name[:-4].split('_')[1]

    # Select the files containing the density values based on actual traffic records
    select_files = [file for file in os.listdir(path_to_fishnet) if file.startswith('official_' + traffic_type) and
                    considered_bandwidth in file and file[:-4].split('_')[3] == unit_size and file.endswith('.shp')]

    assert len(select_files) == 1, "The corresponding official fishnet file is not selected correctly."

    return select_files[0]


def find_related_official_fishnet_month_day(weibo_shapefile_name, fishnet_directory=raster_fishnet):
    """
    Find the corresponding official fishnet files with KDE values to compute hotspot precision
    :param weibo_shapefile_name: the shapefile for Weibo fishnet
    :param fishnet_directory: the path saving the fishnet
    :return: the name of the corresponding official fishnet file
    """
    if 'acc' in weibo_shapefile_name:
        traffic_type = 'acc'
    else:
        traffic_type = 'cgs'

    # Get the spatial unit size and the bandwidth
    unit_size = weibo_shapefile_name[:-4].split('_')[2]
    considered_bandwidth = weibo_shapefile_name[:-4].split('_')[1]

    # Get the time information
    select_month = weibo_shapefile_name[:-4].split('_')[3]
    select_day = weibo_shapefile_name[:-4].split('_')[4]

    # Get the corresponding shapefiles saving the density value based on actual traffic records
    select_files = [file for file in os.listdir(fishnet_directory) if file.startswith('official_' + traffic_type) and
                    considered_bandwidth in file and file[:-4].split('_')[3] == unit_size and file.endswith('.shp') and
                    file[:-4].split('_')[4] == str(select_month) and file[:-4].split('_')[5] == str(select_day)]

    if not select_files:  # If not fishnet is generated on that day
        return None
    else:
        assert len(select_files) == 1, "The corresponding official fishnet file is not selected correctly."
        return select_files[0]


def compute_raster_fishnet(fishnet_filename: str, raster_path_filename: str):
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
    coordinates = [(point.x, point.y) for point in list(fishnet_shape['centroid'])]

    # Compute the raster values
    # The "masked = True" here means that the rasterio package would assign a negative value to the locations
    # which have no raster value in the studied tif file
    raster_vals = np.array([x for x in raster_vals.sample(coordinates, masked=True)])
    raster_vals[raster_vals < 0] = 0

    # Get the final fishnet file with raster values
    fishnet_shape_copy['raster_val'] = raster_vals

    return fishnet_shape_copy


def spatial_join(point_gdf, shape_area):
    """
    Find the tweets posted in one city's open space
    :param point_gdf: the geopandas dataframe saving the tweets
    :param shape_area: the shapefile of a studied area, such as city, open space, etc
    :return: tweets posted in open space
    """
    assert point_gdf.crs == shape_area.crs, 'The coordinate systems do not match!'
    joined_data = gpd.sjoin(left_df=point_gdf, right_df=shape_area, op='within')
    return joined_data


def compute_hotspot_precision(weibo_fishnet, actual_fishnet):
    """
    Compute the hotspot identification precision
    For reference: https://www.mdpi.com/2220-9964/8/8/344
    :param weibo_fishnet: the fishnet to identify hotspot based on traffic-relevant Weibos
    :param actual_fishnet: the fishnet to identify hotspot based on actual traffic records
    :return: the hotspot precision value
    """
    # Make sure the fishnet files save the raster values and have same size
    assert 'raster_val' in weibo_fishnet, "The weibo hotspot fishnet should save raster values"
    assert 'raster_val' in actual_fishnet, "The actual hotspot fishnet should save raster values"
    assert weibo_fishnet.shape[0] == actual_fishnet.shape[0], \
        "The shapefiles should have the same size. But one is {} and another is {}".format(
            weibo_fishnet.shape[0], actual_fishnet.shape[0])
    weibo_fishnet_renamed = weibo_fishnet.rename(columns={'raster_val': "raster_val_weibo"})
    actual_fishnet_renamed = actual_fishnet.rename(columns={'raster_val': "raster_val_actual"})

    # Combine two fishnets
    raster_val_actual = list(actual_fishnet_renamed['raster_val_actual'])
    combined_geo_data = weibo_fishnet_renamed.copy()
    combined_geo_data['raster_val_actual'] = raster_val_actual

    # Get the Weibo hotspot area and the intersect area and return precision metric
    weibo_density_threshold = np.mean(list(weibo_fishnet_renamed['raster_val_weibo'])) + 3 * np.std(
        list(weibo_fishnet_renamed['raster_val_weibo']))
    actual_density_threshold = np.mean(list(actual_fishnet_renamed['raster_val_actual'])) + 3 * np.std(
        list(actual_fishnet_renamed['raster_val_actual']))
    weibo_hotspot = weibo_fishnet_renamed.loc[weibo_fishnet_renamed['raster_val_weibo'] >= weibo_density_threshold]
    weibo_hotspot_area = get_area(weibo_hotspot)
    combined_geo_data_select = combined_geo_data.loc[
        (combined_geo_data['raster_val_weibo'] >= weibo_density_threshold) & (
                combined_geo_data['raster_val_actual'] >= actual_density_threshold)]
    intersect_area = get_area(combined_geo_data_select)

    return intersect_area / weibo_hotspot_area


def compute_false_alarm_miss_detection(whole_area_grid, point_feature, threshold_value, max_density_val):
    """
    Compute the false alarm rate, miss detection rate, TN, FN, FP, and TP
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
        # Compute the false alarm rate and miss detection rate
        miss_detection_rate = 1 - hotspot_counter[True] / whole_area_counter[True]
        false_alarm_rate = hotspot_counter[False] / whole_area_counter[False]

        # Compute for TN, FN, FP, TP
        TN, FN = not_hotspot_counter[False], not_hotspot_counter[True]
        FP, TP = hotspot_counter[False], hotspot_counter[True]
        for_roc_list = [TN, FN, FP, TP]
        print('False Alarm: {}; Miss Detection: {}; TN, FN, FP, TP: {}'.format(
            false_alarm_rate, miss_detection_rate, for_roc_list))
    else:
        false_alarm_rate, miss_detection_rate = 0, 1
        for_roc_list = [whole_area_counter[False], whole_area_counter[True], 0, 0]

    return false_alarm_rate, miss_detection_rate, for_roc_list


def get_area(shape: gpd.GeoDataFrame):
    """
    Get the total area of a shape (in square kilometers)
    :param shape: the shapefile of an area
    :return: the total area of the shapefile
    """
    return round(sum(shape.area) / 10 ** 6, 4)


def main_extract_raster_for_fishnet(for_official: bool = False):
    """
    Main function to compute raster for fishnet polygons
    :param for_official: for official traffic-relevant records or not
    :return: None. The created fishnet is saved to the local directory
    """
    # Create the fishnet file tif list dict
    fishnet_files = [file for file in os.listdir(fishnet_path) if ('shanghai' in file) and (file.endswith('.shp')) and
                     ('label' not in file)]
    print(fishnet_files)
    fishnet_tif_dict = {}
    for file in fishnet_files:
        grid_cell_size = file[:-4].split('_')[2]
        if for_official:
            select_tif_files = [file for file in os.listdir(tif_save_path) if
                                (file.endswith('.tif') and (file.split('_')[3] == grid_cell_size) and
                                 ('official' in file))]
        else:
            select_tif_files = [file for file in os.listdir(tif_save_path) if
                                (file.endswith('.tif') and (file.split('_')[2] == grid_cell_size) and
                                 ('official' not in file))]
        fishnet_tif_dict[file] = select_tif_files
    print(fishnet_tif_dict)

    # Get the raster value for fishnet polygon based on each raster file
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
                                             consider_sent=False, for_official=False):
    """
    Main function to compute raster for fishnet polygons
    :param: save_path: the pate used to save the created fishnet file
    :param: consider_bandwidth: the considered bandwidth length, in other words, search radius (in meter)
    :param consider_unit_size: the considered spatial unit size (in meter)
    :param consider_sent: whether consider the sentiment information or not
    :param for_official: cope with the official traffic events or not
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
                if consider_sent and (not for_official):  # Weibos & consider sentiment
                    select_tif_files_path = os.path.join(tif_save_path, tif_setting, 'with_sent')
                    select_tif_files_one_setting = [file for file in os.listdir(select_tif_files_path) if
                                                    (file.endswith('.tif') and (file.split('_')[2] == grid_cell_size))]
                elif (not consider_sent) and for_official:  # For official records
                    select_tif_files_path = os.path.join(tif_save_path, tif_setting, 'for_official')
                    select_tif_files_one_setting = [file for file in os.listdir(select_tif_files_path) if
                                                    (file.endswith('.tif') and (file.split('_')[3] == grid_cell_size))]
                else:  # Weibos & not consider sentiment
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

            if 'with_sent' in raster_file:
                raster_setting = '_'.join(raster_file.split('_')[1:3])
                raster_file_path = os.path.join(tif_save_path, raster_setting, 'with_sent', raster_file)
            elif 'official' in raster_file:
                raster_setting = '_'.join(raster_file.split('_')[2:4])
                raster_file_path = os.path.join(tif_save_path, raster_setting, 'for_official', raster_file)
            else:
                raster_setting = '_'.join(raster_file.split('_')[1:3])
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
    :param points: the point feature
    :param hotspot_shape: the shapefile for the hotspot area
    :param shanghai_total_area: the total area of Shanghai, in square kilometers
    :return: PAI value based on points and hotspot area; the number of points in the hotspot
    """
    in_hotspot = spatial_join(point_gdf=points, shape_area=hotspot_shape)
    hotspot_area = get_area(hotspot_shape)
    return (in_hotspot.shape[0] / points.shape[0]) / (hotspot_area / shanghai_total_area), in_hotspot.shape[0]


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
        official_shapefiles_for_sent = [find_related_official_fishnet(file) for file in shapefiles_with_sent]
        official_shapefiles_for_without_sent = [find_related_official_fishnet(file) for file in shapefiles_without_sent]
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_accident.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_accident_shanghai.shp'), encoding='utf-8')
    else:
        shapefiles_with_sent = [file for file in os.listdir(raster_fishnet) if
                                file.startswith('cgs') and file.endswith('with_sent_fishnet.shp') and
                                (str(considered_bandwidth) in file)]
        shapefiles_without_sent = [file for file in os.listdir(raster_fishnet) if
                                   file.startswith('cgs') and file.endswith('without_sent_fishnet.shp') and
                                   (str(considered_bandwidth) in file)]
        official_shapefiles_for_sent = [find_related_official_fishnet(file) for file in shapefiles_with_sent]
        official_shapefiles_for_without_sent = [find_related_official_fishnet(file) for file in shapefiles_without_sent]
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_congestion.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_congestion_shanghai.shp'),
                                       encoding='utf-8')

    print('Considered shapefiles with sentiment information: {}'.format(shapefiles_with_sent))
    print('Considered shapefiles without sentiment information: {}'.format(shapefiles_without_sent))

    setting_list, traffic_type_list, bandwidth_list, unit_size_list = [], [], [], []
    consider_sent_list, PAI_weibo_list, PAI_actual_list, hotspot_area_list = [], [], [], []
    weibo_in_hotspot_count_list, actual_in_hotspot_count_list, threshold_value_list = [], [], []
    hotspot_precision_list = []
    false_alarm_rates_weibo, miss_detection_rates_weibo = [], []
    false_alarm_rates_actual, miss_detection_rates_actual = [], []
    TN_weibo_list, FN_weibo_list, FP_weibo_list, TP_weibo_list = [], [], [], []
    TN_actual_list, FN_actual_list, FP_actual_list, TP_actual_list = [], [], [], []

    print('*' * 20)
    print('Coping with the hotspot with sentiment')
    for shapefile, official_shapefile in zip(shapefiles_with_sent, official_shapefiles_for_sent):
        print('Coping with the file: {} & {}'.format(shapefile, official_shapefile))
        consider_fishnet = gpd.read_file(os.path.join(raster_fishnet, shapefile), encoding='utf-8')
        consider_official_fishnet = gpd.read_file(os.path.join(raster_fishnet, official_shapefile), encoding='utf-8')
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
        hotspot_precision_list.append(compute_hotspot_precision(weibo_fishnet=consider_fishnet,
                                                                actual_fishnet=consider_official_fishnet))

    print('*' * 20)
    print('Coping with the hotspot without sentiment')
    for shapefile, official_shapefile in zip(shapefiles_without_sent, official_shapefiles_for_without_sent):
        print('Coping with the file: {} & {}'.format(shapefile, official_shapefile))
        consider_fishnet = gpd.read_file(os.path.join(raster_fishnet, shapefile), encoding='utf-8')
        consider_official_fishnet = gpd.read_file(os.path.join(raster_fishnet, official_shapefile), encoding='utf-8')
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
        hotspot_precision_list.append(compute_hotspot_precision(weibo_fishnet=consider_fishnet,
                                                                actual_fishnet=consider_official_fishnet))

    # Incrementally save the hotspot validation result to the dataframe...
    print('Save the dataframe for hotspot module validation...')
    pai_dataframe = pd.DataFrame()
    pai_dataframe['setting'] = setting_list
    pai_dataframe['traffic_type'] = traffic_type_list
    pai_dataframe['consider_sent'] = consider_sent_list
    pai_dataframe['weibo_in_hotspot'] = weibo_in_hotspot_count_list
    pai_dataframe['actual_in_hotspot'] = actual_in_hotspot_count_list
    pai_dataframe['hotspot_area'] = hotspot_area_list
    pai_dataframe['hotspot_precision'] = hotspot_precision_list
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
        considered_official_fishnet_path = os.path.join(kde_analysis, 'raster_polygons', 'official_acc')
    else:
        weibo_point = gpd.read_file(os.path.join(shapefile_path, 'weibo_congestion.shp'), encoding='utf-8')
        official_point = gpd.read_file(os.path.join(shapefile_path, 'official_congestion_shanghai.shp'),
                                       encoding='utf-8')
        considered_fishnet_path = os.path.join(kde_analysis, 'raster_polygons', 'cgs_2000_260')
        considered_official_fishnet_path = os.path.join(kde_analysis, 'raster_polygons', 'official_cgs')
    if consider_sent:  # consider sentiment or not
        considered_fishnet_shapefiles = [file for file in os.listdir(considered_fishnet_path) if
                                         (file.endswith('.shp')) and ('with_sent' in file)]
    else:
        considered_fishnet_shapefiles = [file for file in os.listdir(considered_fishnet_path) if
                                         (file.endswith('.shp')) and ('without_sent' in file)]
    considered_official_fishnet_shapefiles = [find_related_official_fishnet_month_day(
        file, fishnet_directory=considered_official_fishnet_path) for file in considered_fishnet_shapefiles]
    print('Considered Weibo fishnet files: {}'.format(considered_fishnet_shapefiles))
    print('Considered official fishnet files: {}'.format(considered_official_fishnet_shapefiles))

    # Create the list saving the metrics for hotspot daily validation
    setting_list, traffic_type_list = [], []
    bandwidth_list, unit_size_list = [], []
    month_list, day_list = [], []
    PAI_weibo_list, PAI_actual_list, hotspot_area_list = [], [], []
    weibo_in_hotspot_count_list, actual_in_hotspot_count_list, threshold_value_list = [], [], []
    false_alarm_rates_weibo, miss_detection_rates_weibo = [], []
    false_alarm_rates_actual, miss_detection_rates_actual = [], []
    hotspot_precision_list, considered_days = [], []
    TN_weibo_list, FN_weibo_list, FP_weibo_list, TP_weibo_list = [], [], [], []
    TN_actual_list, FN_actual_list, FP_actual_list, TP_actual_list = [], [], [], []

    # Compute the metrics for hotspot daily validation
    print('*' * 20)
    for shapefile, official_shapefile in zip(considered_fishnet_shapefiles, considered_official_fishnet_shapefiles):
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
        if (select_actual_point.shape[0] > 1) and (not official_shapefile):
            PAI_actual, actual_count_in_hotspot = compute_pai_and_count(points=select_actual_point,
                                                                        hotspot_shape=hotspot_area)
            consider_official_fishnet = gpd.read_file(
                os.path.join(considered_official_fishnet_path, official_shapefile),
                encoding='utf-8')
            hotspot_precision_list.append(compute_hotspot_precision(weibo_fishnet=consider_fishnet,
                                                                    actual_fishnet=consider_official_fishnet))
            considered_days.append(True)
        else:
            actual_count_in_hotspot = 0
            PAI_actual = 0
            hotspot_precision_list.append(0)
            considered_days.append(False)
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
            false_rate_actual = hotspot_area.shape[0] / consider_fishnet.shape[0]
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
    pai_dataframe['hotspot_precision'] = hotspot_precision_list
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
    # Get the fishnet having raster values
    main_extract_raster_for_fishnet(for_official=False)
    main_extract_raster_for_fishnet(for_official=True)

    # Compare the kde hotspots based on different threshold
    starting_time = time.time()
    build_pai_compare_dataframe_multiple_thresholds(traffic_type='acc', threshold_num=70)
    build_pai_compare_dataframe_multiple_thresholds(traffic_type='cgs', threshold_num=70)
    build_pai_compare_dataframe(traffic_type='acc', considered_bandwidth=2000)
    build_pai_compare_dataframe(traffic_type='cgs', considered_bandwidth=2000)

    # Evaluate the performance of the selected hotspot identification module from people's perspectives
    build_pai_dataframe_for_each_day(traffic_type='acc', consider_sent=False)
    build_pai_dataframe_for_each_day(traffic_type='cgs', consider_sent=False)
    ending_time = time.time()

    print('Total time: {}h'.format(round((ending_time - starting_time) / 3600, 2)))