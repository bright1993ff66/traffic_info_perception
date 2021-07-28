import os
import time
import numpy as np
import arcpy
from arcpy import env
from arcpy.sa import *
from matplotlib import pyplot as plt

from data_paths import project_path, figures_path

# Set environment settings
env.workspace = "D:\Projects\Traffic_info_perception\gis_analysis_project\kde_analysis\shapefile"
env.overwriteOutput = True

shapefile_path = os.path.join(project_path, 'gis_analysis_project')
kde_analysis = os.path.join(shapefile_path, 'kde_analysis')
kde_analysis_on_each_day = os.path.join(kde_analysis, 'kde_on_each_day')
save_path = os.path.join(kde_analysis, 'sensitivity')
tif_save_path = os.path.join(save_path, 'tif_files')
raster_table_save_path = os.path.join(save_path, 'raster_table')
raster_table_with_sent = os.path.join(raster_table_save_path, 'with_sent')
raster_table_without_sent = os.path.join(raster_table_save_path, 'without_sent')

polygon_save_path = os.path.join(save_path, 'polygon_path')
polygon_temp_path = os.path.join(polygon_save_path, 'temp')
polygon_units_path = os.path.join(polygon_save_path, 'polygons')
polygon_sent_temp = os.path.join(polygon_save_path, 'sent_temp')
polygon_without_sent_temp = os.path.join(polygon_save_path, 'without_sent_temp')
polygon_path_with_sent = os.path.join(polygon_save_path, 'with_sent')
polygon_path_without_sent = os.path.join(polygon_save_path, 'without_sent')

array_value_path = os.path.join(save_path, 'density_vals')
hotspot_figure_path = os.path.join(save_path, 'find_hotspot_figures')

bandwidths = [1000, 2000, 3000]
spatial_unit_sizes = [200, 260, 300]
std_sizes_check = [2, 3]


def get_kde_each_day(inFeature_filename, consider_sent, unit_size, bandwidth, months_list, days_list):
    """
    Compute kernel density based on points using ArcMap. For reference:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/kernel-density.htm
    :param inFeature_filename: the name of the input point feature
    :param consider_sent: consider the sentiment or not
    :param unit_size: the unit size for kde analysis
    :param bandwidth: the bandwidth for kde analysis
    :param months_list: a list for the considered months
    :param days_list: a list for the considered days
    :return: None. The computed density values are saved to local directory as .tif files
    """
    assert 'accident' in inFeature_filename or 'congestion' in inFeature_filename, \
        "The input file should include either accident or congestion"

    print('Coping with the file: {}'.format(inFeature_filename))
    print('Considering the sentiment: {}'.format(consider_sent))

    # Check the traffic type to run
    if 'acc' in inFeature_filename:
        traffic_type = 'acc'
    else:
        traffic_type = 'cgs'

    # Consider sentiment or not
    if consider_sent:
        populationField = 'sent_val'
    else:
        populationField = None

    # In Raster value
    inRaster=100

    # Compute the kernel density values
    for month in months_list:
        for day in days_list:
            print('Coping with the time: month: {}, day: {}'.format(month, day))
            env.workspace = os.path.join(kde_analysis, "shapefile")
            arcpy.MakeFeatureLayer_management(inFeature_filename, '{}_point_lyr'.format(traffic_type))
            selection_statement = """ "month" = {} AND "day" = {} """.format(month, day)
            selected_points = arcpy.SelectLayerByAttribute_management(
                in_layer_or_view='{}_point_lyr'.format(traffic_type),
                selection_type='NEW_SELECTION',
                where_clause=selection_statement)
            row_count = int(arcpy.GetCount_management(selected_points).getOutput(0))
            if row_count > 0:
                output_density = KernelDensity(
                    in_features=selected_points,
                    population_field=populationField,
                    cell_size=unit_size,
                    search_radius=bandwidth,
                    area_unit_scale_factor='SQUARE_KILOMETERS',
                    out_cell_values='DENSITIES',
                    method='PLANAR')
                output_density_100 = inRaster * output_density

                if consider_sent:
                    arcpy.CopyRaster_management(in_raster=output_density_100,
                                                out_rasterdataset=os.path.join(
                                                    kde_analysis_on_each_day, 'raster_table',
                                                    '{}_{}_{}_s'.format(traffic_type[0], month, day)),
                                                pixel_type='32_BIT_SIGNED')
                    output_density.save(os.path.join(kde_analysis_on_each_day, 'raster_tifs',
                                                     '{}_{}_{}_with_sent.tif'.format(traffic_type, month, day)))
                else:
                    arcpy.CopyRaster_management(in_raster=output_density_100,
                                                out_rasterdataset=os.path.join(
                                                    kde_analysis_on_each_day, 'raster_table',
                                                    '{}_{}_{}_ws'.format(traffic_type[0], month, day)),
                                                pixel_type='32_BIT_SIGNED')
                    output_density.save(os.path.join(kde_analysis_on_each_day, 'raster_tifs',
                                                     '{}_{}_{}_without_sent.tif'.format(traffic_type, month, day)))
            else:
                print('We do not have {} Weibos posted on 2012-{}-{}'.format(traffic_type, month, day))


def compare_kernel_density(inFeature_filename, consider_sent, bandwidth_list, unit_size_list,
                           standard_deviation_sizes, aggregate_polygon=True):
    """
    Compute kernel density based on points using ArcMap. For reference:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/kernel-density.htm
    :param inFeature_filename: the name of the input point feature
    :param consider_sent: consider the sentiment or not
    :param bandwidth_list: a list saving the candidate bandwidths
    :param unit_size_list: a list saving the spatial unit sizes
    :param standard_deviation_sizes: a list saving the considered standard deviation sizes
    :param aggregate_polygon: aggregate the spatial units or not
    :return: None. The computed density values are saved to local directory as .tif files
    """
    assert 'accident' in inFeature_filename or 'congestion' in inFeature_filename, \
        "The input file should include either accident or congestion"

    print('Coping with the file: {}'.format(inFeature_filename))
    print('Considering the sentiment: {}'.format(consider_sent))

    # Check the traffic type to run
    if 'acc' in inFeature_filename:
        traffic_type = 'acc'
    else:
        traffic_type = 'cgs'

    # Consider sentiment or not
    if consider_sent:
        populationField = 'sent_val'
    else:
        populationField = None

    # In Raster value
    inRaster=100

    # Compute the kernel density values
    for bandwidth in bandwidth_list:
        for unit_size in unit_size_list:
            for std_size in standard_deviation_sizes:
                print('Coping with the setting: bandwidth: {}, spatial unit size: {}, std size: {}'.format(
                    bandwidth, unit_size, std_size))
                env.workspace = os.path.join(kde_analysis, "shapefile")
                output_density = KernelDensity(
                    in_features=inFeature_filename,
                    population_field=populationField,
                    cell_size=unit_size,
                    search_radius=bandwidth,
                    area_unit_scale_factor='SQUARE_KILOMETERS',
                    out_cell_values='DENSITIES',
                    method='PLANAR')
                output_density_100 = inRaster * output_density

                if consider_sent:
                    output_density.save(os.path.join(tif_save_path, '{}_{}_{}_{}_sent.tif'.format(
                        traffic_type, bandwidth, unit_size, std_size)))
                    density_values = arcpy.RasterToNumPyArray(output_density)
                    density_values_ravel = density_values.ravel()
                    threshold_value = np.mean(density_values_ravel) + std_size * np.std(density_values_ravel)
                    print('The threshold value is set to: {}'.format(threshold_value))
                    np.save(os.path.join(array_value_path, '{}_{}_{}_{}_sent.npy'.format(
                        traffic_type, bandwidth, unit_size, std_size)), density_values)
                    try:
                        arcpy.CopyRaster_management(in_raster=output_density_100,
                                                    out_rasterdataset=os.path.join(
                                                        raster_table_with_sent, '{}_{}_{}_{}'.format(traffic_type[0],
                                                                                                          bandwidth,
                                                                                                          unit_size,
                                                                                                          std_size)),
                                                    pixel_type='32_BIT_SIGNED')
                        env.workspace = raster_table_with_sent
                        raster_layer = arcpy.Raster('{}_{}_{}_{}'.format(traffic_type[0],
                                                                         bandwidth,
                                                                         unit_size,
                                                                         std_size))
                        temp_polygon_save = os.path.join(polygon_sent_temp,
                                                         '{}_{}_{}_{}.shp'.format(traffic_type[0],
                                                                         bandwidth,
                                                                         unit_size,
                                                                         std_size))
                        arcpy.RasterToPolygon_conversion(raster_layer, temp_polygon_save,
                                                         "NO_SIMPLIFY")
                        polygon_file = arcpy.MakeFeatureLayer_management(temp_polygon_save)
                        select_layer = arcpy.SelectLayerByAttribute_management(polygon_file,
                                                                               "NEW_SELECTION",
                                                                               '"GRIDCODE" > {}'.format(
                                                                                   int(round(threshold_value * 100,
                                                                                             0))))
                        save_file = os.path.join(polygon_path_with_sent, '{}_{}_{}_{}.shp'.format(traffic_type[0],
                                                                                      bandwidth,
                                                                                      unit_size,
                                                                                      std_size))
                        if aggregate_polygon:
                            arcpy.AggregatePolygons_cartography(select_layer, save_file, "{} Meters".format(unit_size))
                        else:
                            arcpy.CopyFeatures_management(select_layer, save_file)
                    except:
                        print("Copy Raster example failed.")
                        print(arcpy.GetMessages())
                else:
                    output_density.save(os.path.join(tif_save_path, '{}_{}_{}_{}_without_sent.tif'.format(
                        traffic_type, bandwidth, unit_size, std_size)))
                    density_values = arcpy.RasterToNumPyArray(output_density)
                    density_values_ravel = density_values.ravel()
                    threshold_value = np.mean(density_values_ravel) + std_size * np.std(density_values_ravel)
                    print('The threshold value is set to: {}'.format(threshold_value))
                    np.save(os.path.join(array_value_path, '{}_{}_{}_{}_without_sent.npy'.format(
                        traffic_type, bandwidth, unit_size, std_size)), density_values)
                    try:
                        arcpy.CopyRaster_management(in_raster=output_density_100,
                                                    out_rasterdataset=os.path.join(
                                                        raster_table_without_sent, '{}_{}_{}_{}'.format(
                                                            traffic_type[0], bandwidth, unit_size, std_size)),
                                                    pixel_type='32_BIT_SIGNED')
                        env.workspace = raster_table_without_sent
                        raster_layer = arcpy.Raster('{}_{}_{}_{}'.format(traffic_type[0],
                                                 bandwidth,
                                                 unit_size,
                                                 std_size))
                        temp_polygon_save = os.path.join(polygon_without_sent_temp,
                                                         '{}_{}_{}_{}.shp'.format(traffic_type[0],
                                                                                  bandwidth,
                                                                                  unit_size,
                                                                                  std_size))
                        arcpy.RasterToPolygon_conversion(raster_layer, temp_polygon_save,
                                                         "NO_SIMPLIFY")
                        polygon_file = arcpy.MakeFeatureLayer_management(temp_polygon_save)
                        select_layer = arcpy.SelectLayerByAttribute_management(polygon_file,
                                                                               "NEW_SELECTION",
                                                                               '"GRIDCODE" > {}'.format(
                                                                                   int(round(threshold_value * 100,
                                                                                             0))))
                        save_file = os.path.join(polygon_path_without_sent, '{}_{}_{}_{}.shp'.format(traffic_type[0],
                                                                                                  bandwidth,
                                                                                                  unit_size,
                                                                                                  std_size))
                        if aggregate_polygon:
                            arcpy.AggregatePolygons_cartography(select_layer, save_file, "{} Meters".format(unit_size))
                        else:
                            arcpy.CopyFeatures_management(select_layer, save_file)
                    except:
                        print("Copy Raster example failed.")
                        print(arcpy.GetMessages())


                print('The mean is: {}'.format(output_density.mean))
                print('The std is: {}'.format(output_density.standardDeviation))
                print('The minimum is: {}'.format(output_density.minimum))
                print('The maximum is: {}'.format(output_density.maximum))


def compare_kernel_for_roc(inFeature_points, consider_sent, bandwidth_list, unit_size_list,
                       aggregate_polygon=True):
    """
    Compute kernel density based on points using ArcMap. For reference:
    https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/kernel-density.htm
    :param inFeature_points: the name of the input point feature
    :param consider_sent: consider the sentiment or not
    :param bandwidth_list: a list saving the candidate bandwidths
    :param unit_size_list: a list saving the spatial unit sizes
    :param aggregate_polygon: aggregate the spatial units or not
    :return: None. The computed density values are saved to local directory as .tif files
    """
    assert 'accident' in inFeature_points or 'congestion' in inFeature_points, \
        "The input file should include either accident or congestion"

    print('Coping with the point feature: {}'.format(inFeature_points))
    print('Considering the sentiment or not: {}'.format(consider_sent))

    # Check the traffic type to run
    if 'acc' in inFeature_points:
        traffic_type = 'acc'
    else:
        traffic_type = 'cgs'

    # Consider sentiment or not
    if consider_sent:
        populationField = 'sent_val'
    else:
        populationField = None

    # In Raster value
    inRaster=100

    # Compute the kernel density values, tune the bandwidth and unit size
    for bandwidth in bandwidth_list:
        for unit_size in unit_size_list:
            print('Coping with the setting: bandwidth: {}, spatial unit size: {}'.format(
                bandwidth, unit_size))
            env.workspace = os.path.join(kde_analysis, "shapefile")
            output_density = KernelDensity(
                in_features=inFeature_points,
                population_field=populationField,
                cell_size=unit_size,
                search_radius=bandwidth,
                area_unit_scale_factor='SQUARE_KILOMETERS',
                out_cell_values='DENSITIES',
                method='PLANAR')
            output_density_100 = inRaster * output_density
            # Save the computed kernel density raster
            # Save the 100 * raster val into 32 bit signed in order to visualize it in arcmap
            if consider_sent:
                arcpy.CopyRaster_management(in_raster=output_density_100,
                                            out_rasterdataset=os.path.join(
                                                raster_table_save_path, '{}_{}_{}_s'.format(traffic_type[0],
                                                                                          bandwidth,
                                                                                          unit_size)),
                                            pixel_type='32_BIT_SIGNED')
                output_density.save(os.path.join(tif_save_path, '{}_{}_{}_with_sent.tif'.format(
                    traffic_type, bandwidth, unit_size)))
            else:
                arcpy.CopyRaster_management(in_raster=output_density_100,
                                            out_rasterdataset=os.path.join(
                                                raster_table_save_path, '{}_{}_{}_ws'.format(traffic_type[0],
                                                                                            bandwidth,
                                                                                            unit_size)),
                                            pixel_type='32_BIT_SIGNED')
                output_density.save(os.path.join(tif_save_path, '{}_{}_{}_without_sent.tif'.format(
                    traffic_type, bandwidth, unit_size)))

            # # Generate 100 density value choices between mean + 2*std and mean + 3*std
            # density_values = arcpy.RasterToNumPyArray(output_density)
            # density_values_ravel = density_values.ravel()
            # density_val_mean, density_val_std = np.mean(density_values_ravel), np.std(density_values_ravel)
            # density_threshold_low = density_val_mean + 2 * density_val_std
            # density_thresold_up = density_val_mean + 3 * density_val_std
            # threshold_choices = np.linspace(density_threshold_low, density_thresold_up, 100)
            # # Save the 100 * Raster to Polygon
            # env.workspace = raster_table_save_path
            # if consider_sent:
            #     np.save(os.path.join(array_value_path, '{}_{}_{}_sent.npy'.format(
            #         traffic_type, bandwidth, unit_size)), density_values)
            #     raster_layer = arcpy.Raster('{}_{}_{}_s'.format(traffic_type[0], bandwidth, unit_size))
            #     temp_polygon_save = os.path.join(polygon_temp_path,
            #                                      '{}_{}_{}_sent.shp'.format(traffic_type[0],
            #                                                                 bandwidth,
            #                                                                 unit_size))
            # else:
            #     np.save(os.path.join(array_value_path, '{}_{}_{}_without_sent.npy'.format(
            #         traffic_type, bandwidth, unit_size)), density_values)
            #     raster_layer = arcpy.Raster('{}_{}_{}_ws'.format(traffic_type[0], bandwidth, unit_size))
            #     temp_polygon_save = os.path.join(polygon_temp_path,
            #                                      '{}_{}_{}_without_sent.shp'.format(traffic_type[0],
            #                                                                 bandwidth,
            #                                                                 unit_size))
            # arcpy.RasterToPolygon_conversion(raster_layer, temp_polygon_save,
            #                                  "NO_SIMPLIFY")
            # print('The mean is: {}'.format(output_density.mean))
            # print('The std is: {}'.format(output_density.standardDeviation))
            # print('The minimum is: {}'.format(output_density.minimum))
            # print('The maximum is: {}'.format(output_density.maximum))
            #
            # # Load the saved polygon and create the hotspot area based on one threshold value
            # polygon_file = arcpy.MakeFeatureLayer_management(temp_polygon_save)
            # print('Start getting the hotspot area with different threshold values...')
            # for threshold_value in threshold_choices:
            #     print('The threshold value is set to: {}'.format(threshold_value))
            #     # try:
            #     select_layer = arcpy.SelectLayerByAttribute_management(polygon_file,
            #                                                            "NEW_SELECTION",
            #                                                            '"GRIDCODE" > {}'.format(
            #                                                                int(round(threshold_value * 100, 0))))
            #     if consider_sent:
            #         save_file = os.path.join(polygon_units_path, '{}_{}_{}_{}_sent.shp'.format(traffic_type[0],
            #                                                                                   bandwidth,
            #                                                                                   unit_size, int(round(
            #                 threshold_value * 100,
            #                 0))))
            #     else:
            #         save_file = os.path.join(polygon_units_path, '{}_{}_{}_{}_without_sent.shp'.format(traffic_type[0],
            #                                                                                    bandwidth,
            #                                                                                    unit_size, int(round(
            #                 threshold_value * 100,
            #                 0))))
            #     if aggregate_polygon:
            #         arcpy.AggregatePolygons_cartography(select_layer, save_file, "{} Meters".format(unit_size))
            #     else:
            #         arcpy.CopyFeatures_management(select_layer, save_file)
            #     # except:
            #     #     print("Select raster based on threshold value failed.")
            #     #     print(arcpy.GetMessages())


if __name__ == '__main__':
    # compare_kernel_density(inFeature_filename='weibo_accident.shp', consider_sent=True,
    #                        bandwidth_list=bandwidths, unit_size_list=spatial_unit_sizes,
    #                        standard_deviation_sizes=std_sizes_check, aggregate_polygon=False)
    # compare_kernel_density(inFeature_filename='weibo_accident.shp', consider_sent=False,
    #                        bandwidth_list=bandwidths, unit_size_list=spatial_unit_sizes,
    #                        standard_deviation_sizes=std_sizes_check, aggregate_polygon=False)
    # compare_kernel_density(inFeature_filename='weibo_congestion.shp', consider_sent=True,
    #                        bandwidth_list=bandwidths, unit_size_list=spatial_unit_sizes,
    #                        standard_deviation_sizes=std_sizes_check, aggregate_polygon=False)
    # compare_kernel_density(inFeature_filename='weibo_congestion.shp', consider_sent=False,
    #                        bandwidth_list=bandwidths, unit_size_list=spatial_unit_sizes,
    #                        standard_deviation_sizes=std_sizes_check, aggregate_polygon=False)
    considered_months_list = [6, 7, 8]
    considered_days_list = list(range(1, 32))
    compare_kernel_for_roc(inFeature_points='weibo_accident.shp', consider_sent=True,
                           bandwidth_list=bandwidths, unit_size_list=spatial_unit_sizes,
                           aggregate_polygon=False)
    compare_kernel_for_roc(inFeature_points='weibo_accident.shp', consider_sent=False,
                           bandwidth_list=bandwidths, unit_size_list=spatial_unit_sizes,
                           aggregate_polygon=False)
    compare_kernel_for_roc(inFeature_points='weibo_congestion.shp', consider_sent=True,
                           bandwidth_list=bandwidths, unit_size_list=spatial_unit_sizes,
                           aggregate_polygon=False)
    compare_kernel_for_roc(inFeature_points='weibo_congestion.shp', consider_sent=False,
                           bandwidth_list=bandwidths, unit_size_list=spatial_unit_sizes,
                           aggregate_polygon=False)
