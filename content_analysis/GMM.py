import pandas as pd
import os
import numpy as np
import random
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from collections import Counter
import geopandas as gpd
import itertools

from utils import random_seed, count_positive_neutral_negative
from data_paths import hotspot_text_path, shapefile_path, figures_path, kde_analysis
from location_analysis import create_sentiment_against_density_data
from visualizations import sentiment_against_density_plot

# Set the font type
font = {'family': 'serif'}
mpl.rc('font', **font)


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_bic(weibo_dataframe: pd.DataFrame, save_filename: str):
    """
    Plot the BIC to select the best number of components for the Gaussian Mixture Model (GMM)
    :param weibo_dataframe: a pandas dataframe saving the locations of Weibo based traffic events.
    :param save_filename: the name of the saved file for the BIC plot
    :return: A bic figure using GMM is saved to local directory
    """
    assert 'acc' in save_filename or 'cgs' in save_filename, "The save filename is not given properly"
    if 'acc' in save_filename:
        print('Computing the BIC for the accident relevant Weibos...')
    else:
        print('Computing the BIC for the congestion relevant Weibos...')
    location_array = np.array(
        [[val1, val2] for val1, val2 in zip(weibo_dataframe['lon_val'], weibo_dataframe['lat_val'])])
    n_components = np.arange(5, 100)  # Try components from 5 to 99
    models = [GaussianMixture(n, covariance_type='full', random_state=random_seed).fit(location_array)
              for n in n_components]
    figure, axis = plt.subplots(1, 1, figsize=(10, 8))
    bic_values = [m.bic(location_array) for m in models]
    if 'acc' in save_filename:
        axis.plot(n_components, bic_values, color='blue')
        axis.axvline(54, color='black', linestyle='--', alpha=0.5)
        axis.text(58, -12000, 'Number of Clusters We Select: {}'.format(54), color='black')
    else:
        axis.plot(n_components, bic_values, color='orange')
        axis.axvline(32, color='black', linestyle='--', alpha=0.5)
        axis.text(36, -8500, 'Number of Clusters We Select: {}'.format(32), color='black')
    axis.set_xticks(np.arange(5, 100, 10))
    axis.xaxis.set_tick_params(size=5)
    axis.set_xlabel('# of Predefined GMM Clusters')
    if 'acc' in save_filename:
        axis.set_title('GMM Parameter Selection - Accidents')
    else:
        axis.set_title('GMM Parameter Selection - Congestions')
    axis.set_ylabel('BIC Value')

    # Delete unnecessary spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    figure.savefig(os.path.join(figures_path, save_filename))


def plot_gmm(gmm, X, label_threshold, label_counter, label=True, ax=None):
    """
    Plot the gmm clustering result based on traffic event locations
    :param gmm: a Gaussian Mixture Model
    :param X: a numpy array saving the locations of traffic accidents or congestions
    :param label_threshold: a threshold value used to choose the GMM cluster
    :param label_counter: a counter of each GMM label based on number of Weibos and their reposts
    :param label: Whether label a point with its gmm label or not
    :param ax: a matplotlib axis
    :return:
    """
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    select_label_info = np.array(
        [[index, label] for index, label in enumerate(labels) if label_counter[label] > label_threshold])
    select_locations = X[select_label_info[:, 0]]
    select_labels = select_label_info[:, 1]
    print("The gmm labels we select are: ")
    print(set(select_labels))
    if label:
        scatter = ax.scatter(select_locations[:, 0], select_locations[:, 1], c=select_labels, s=5, cmap='Paired',
                             zorder=2, alpha=0.5, edgecolors='none')
    else:
        scatter = ax.scatter(select_locations[:, 0], select_locations[:, 1], s=5, zorder=2, alpha=0.5,
                             edgecolors='none')
    # Set the axis equally
    ax.axis('equal')

    # Produce a legend with the unique colors from the scatter
    legend_scatter = ax.legend(handles=scatter.legend_elements()[0], labels=list(set(select_labels.tolist())),
                               loc="lower left", title="GMM Cluster Labels")
    ax.add_artist(legend_scatter)

    # Create ellipses
    select_means = []
    w_factor = 0.2 / gmm.weights_.max()
    for label, pos, covar, w in zip(list(range(len(gmm.means_))), gmm.means_, gmm.covariances_, gmm.weights_):
        if label in set(select_labels):
            draw_ellipse(pos, covar, alpha=w * w_factor, color='grey', ax=ax)
            select_means.append(pos)
    means_array = np.array(select_means)
    # ax.plot(means_array[:, 0], means_array[:, 1], '^', color='black', markersize=5)
    return select_labels


def plot_gmm_with_area(n_gmm_components, weibo_dataframe, actual_dataframe, area_shape,
                       cluster_sentiment_filename, gmm_cluster_map_filename, weibo_repost_count_threshold=50,
                       consider_sentiment=True, plot_kde_hotspot=True):
    """
    Create the scatter plot for the sentiment and number of Weibos and their reposts for each GMM cluster
    :param n_gmm_components: number of Gaussian Mixture Model clusters
    :param weibo_dataframe: a pandas dataframe saving the accident or congestion Weibos
    :param actual_dataframe: a pandas dataframe saving the actual accident or congestion records
    :param area_shape: the shapefile of the area, in our case, Shanghai
    :param cluster_sentiment_filename: the saved filename saving the sentiment index and number of Weibos and their
    reposts for each cluster
    :param gmm_cluster_map_filename: the saved filename saving the gmm clusters on the map
    :param weibo_repost_count_threshold: The number of Weibos and their reposts
    :param consider_sentiment: only consider the negative Weibos or not
    :param plot_kde_hotspot: whether add the previous kde hotspot result to the figure
    :return: The sentiment against the weibo repost count plot; The spatial distribution of gmm clusters on the map
    """
    assert 'acc' in cluster_sentiment_filename or 'cgs' in cluster_sentiment_filename, \
        "The filename should be set properly, either contain 'acc' or 'cgs'."
    assert 'acc' in gmm_cluster_map_filename or 'cgs' in gmm_cluster_map_filename, \
        "The filename should be set properly, either contain 'acc' or 'cgs'."
    if consider_sentiment:
        weibo_dataframe_select = weibo_dataframe.copy()
    else:
        weibo_dataframe_select = weibo_dataframe.loc[weibo_dataframe['sent_val'] == 0]
    selected_location_array = np.array(
        [[val1, val2] for val1, val2 in zip(weibo_dataframe_select['lon_val'], weibo_dataframe_select['lat_val'])])
    actual_location_array = np.array(
        [[val1, val2] for val1, val2 in zip(actual_dataframe['loc_lon'], actual_dataframe['loc_lat'])])

    gmm = GaussianMixture(n_components=n_gmm_components, covariance_type='full', random_state=random_seed)
    labels = gmm.fit(selected_location_array).predict(selected_location_array)
    weibo_dataframe_select_copy = weibo_dataframe_select.copy()
    weibo_dataframe_select_copy['gmm_labels'] = labels
    select_columns = ['index_val', 'author_id', 'weibo_id', 'created_at', 'text', 'lat', 'lon', 'retweeters',
                      'retweete_1', 'local_time', 'year', 'month', 'traffic_we', 'traffic_re', 'sent_weibo',
                      'sent_repos', 'Name', 'datatype', 'traffic_ty', 'hotspot_id', 'Avg_Dense', 'Area',
                      'gmm_labels']
    rename_dict = {'traffic_we': 'traffic_weibo', 'traffic_re': 'traffic_repost', 'retweete_1': 'retweeters_text'}
    weibo_data_select_renamed = weibo_dataframe_select_copy[select_columns].rename(columns=rename_dict)
    if 'acc' in gmm_cluster_map_filename:
        weibo_data_select_renamed.to_csv(os.path.join(hotspot_text_path,
                                                      'weibo_acc_hotspot_join_with_gmm.csv'), encoding='utf-8')
    else:
        weibo_data_select_renamed.to_csv(os.path.join(hotspot_text_path,
                                                      'weibo_cgs_hotspot_join_with_gmm.csv'), encoding='utf-8')

    # Count the number of Weibos and their reposts for each gmm label
    label_counter = {}
    for label, dataframe in weibo_data_select_renamed.groupby('gmm_labels'):
        pos_count, neutral_count, neg_count = count_positive_neutral_negative(
            dataframe, repost_column='retweeters_text')
        label_counter[label] = pos_count + neutral_count + neg_count
    area_shape_transformed = area_shape.to_crs(epsg=4326)
    if plot_kde_hotspot and 'acc' in gmm_cluster_map_filename:
        hotspot_shape = gpd.read_file(os.path.join(kde_analysis, 'shapefile', 'default',
                                                   'acc_hotspot_blocks_merged.shp'))
        hotspot_shape_transformed = hotspot_shape.to_crs(epsg=4326)
    elif plot_kde_hotspot and 'cgs' in gmm_cluster_map_filename:
        hotspot_shape = gpd.read_file(os.path.join(kde_analysis, 'shapefile', 'default',
                                                   'cgs_hotspot_blocks_merged.shp'))
        hotspot_shape_transformed = hotspot_shape.to_crs(epsg=4326)
    else:
        hotspot_shape_transformed = None
    # Create the figures
    figure, axis = plt.subplots(1, 1, figsize=(15, 15), dpi=300)
    sentiment_against_gmm_cluster = create_sentiment_against_density_data(weibo_dataframe=weibo_data_select_renamed,
                                                                          hotspot_id_colname='gmm_labels',
                                                                          repost_colname='retweeters_text')
    sentiment_against_density_plot(hotspot_sent_density_data=sentiment_against_gmm_cluster,
                                   save_filename=cluster_sentiment_filename, dot_annotate=True)
    plot_gmm(gmm, X=selected_location_array, ax=axis, label_threshold=weibo_repost_count_threshold,
             label_counter=label_counter, label=True)
    if 'acc' in gmm_cluster_map_filename:
        axis.plot(actual_location_array[:, 0], actual_location_array[:, 1], 'o', color='black', markersize=2,
                  alpha=0.7, label='Actual Accidents')
    else:
        axis.plot(actual_location_array[:, 0], actual_location_array[:, 1], 'o', color='black', markersize=2,
                  alpha=0.7, label='Actual Congestions')
    area_shape_transformed.plot(ax=axis, color='white', edgecolor='black', legend=True)
    if plot_kde_hotspot:
        hotspot_shape_transformed.plot(ax=axis, color='orange', edgecolor='black', alpha=0.5,
                                                           label='Previous KDE Hotspot', legend=True)
        # legend_kde_hotspot = axis.legend(handles=hotspot_shape_map.legend_elements()[0],
        #                                  labels="KDE Hotspot Area",
        #                                  loc="lower right")
        # axis.add_artist(legend_kde_hotspot)
    # Delete unnecessary spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.legend(fontsize=20)
    axis.set_xlabel('Longitude', size=20)
    axis.set_ylabel('Latitude', size=20)
    figure.savefig(os.path.join(figures_path, gmm_cluster_map_filename), bbox_inches='tight')


if __name__ == '__main__':
    weibo_acc = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_acc_hotspot_district_join.txt'), index_col=0,
                            encoding='utf-8')
    weibo_cgs = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_cgs_hotspot_district_join.txt'), index_col=0,
                            encoding='utf-8')
    official_acc = pd.read_csv(os.path.join(shapefile_path, 'overall', 'official_accident_shanghai.txt'), index_col=0,
                               encoding='utf-8')
    official_cgs = pd.read_csv(os.path.join(shapefile_path, 'overall', 'official_congestion_shanghai.txt'), index_col=0,
                               encoding='utf-8')
    shanghai_shape = gpd.read_file(os.path.join(shapefile_path, 'overall', 'shanghai_proj_utm_51.shp'))
    acc_hotspot_shape = gpd.read_file(os.path.join(kde_analysis, 'shapefile', 'default',
                                                   'acc_hotspot_blocks_merged.shp'))
    cgs_hotspot_shape = gpd.read_file(os.path.join(kde_analysis, 'shapefile', 'with_sent',
                                                   'cgs_hotspot_blocks_merged.shp'))
    # plot_bic(weibo_dataframe=weibo_acc, save_filename='acc_bic.png')
    # plot_bic(weibo_dataframe=weibo_cgs, save_filename='cgs_bic.png')
    plot_gmm_with_area(n_gmm_components=54, weibo_dataframe=weibo_acc, area_shape=shanghai_shape,
                       weibo_repost_count_threshold=30,
                       cluster_sentiment_filename='acc_gmm_cluster_sentiment.png',
                       gmm_cluster_map_filename='acc_gmm_cluster_map_without_sent.png',
                       consider_sentiment=False, plot_kde_hotspot=True,
                       actual_dataframe=official_cgs)
    plot_gmm_with_area(n_gmm_components=32, weibo_dataframe=weibo_cgs, area_shape=shanghai_shape,
                       weibo_repost_count_threshold=60,
                       cluster_sentiment_filename='cgs_gmm_cluster_sentiment.png',
                       gmm_cluster_map_filename='cgs_gmm_cluster_map_with_sent.png',
                       plot_kde_hotspot=True, consider_sentiment=True,
                       actual_dataframe=official_acc)
