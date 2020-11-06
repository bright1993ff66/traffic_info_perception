import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from data_paths import figures_path, weibo_data_path


def create_dendrogram(values: np.ndarray, label_array: np.ndarray, linkage_method: str, savefig_name: str):
    """
    Create the hierarchical clustering dendrogram
    Reference: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
    Scipy reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    :param values: a numpy array saves the values
    :param label_array: the label array for the dendrogram
    :param savefig_name: the filename saved to the local
    :return:
    """
    assert linkage_method in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'], \
        'The method string is not available.'
    linked = linkage(values, method=linkage_method)
    figure, axis = plt.subplots(1, 1, figsize=(10, 8))
    dendrogram(linked,
               orientation='top',
               labels=label_array,
               distance_sort='descending',
               show_leaf_counts=True, ax=axis)
    plt.xticks(rotation=45)
    figure.savefig(os.path.join(figures_path, savefig_name))
    plt.show()


def create_cluster_main(english_xticks, saved_filename):
    """
    Create the cluster for districts
    :param english_xticks: the xticks in English
    :param saved_filename: the saved fig filename
    :return:
    """
    district_count = pd.read_csv(os.path.join(weibo_data_path, 'district_count.csv'), index_col=0, encoding='utf-8')
    district_sent = pd.read_csv(os.path.join(weibo_data_path, 'district_sent.csv'), index_col=0, encoding='utf-8')
    merged_data = district_count.merge(district_sent, left_on=['districts'], right_on=['district_name'])
    # Here we only consider the districts with more than 100 traffic Weibos
    data_for_cluster = merged_data.loc[merged_data['count'] > 100][['districts', 'count', 'sent_index']]
    print('The order of the Chinese districts: {}'.format(list(data_for_cluster['districts'])))
    print('The corresponding English translation: {}'.format(english_xticks))
    values = data_for_cluster[['count', 'sent_index']].values
    create_dendrogram(values=values, label_array=english_xticks, savefig_name=saved_filename,
                      linkage_method='ward')


if __name__ == '__main__':
    english_labels = ['Pudong', 'Zhabei', 'Xuhui', 'Huangpu', 'Yangpu', 'Changning', 'Putuo', 'Minhang',
                      'Hongkou', 'Baoshan', 'Jiading' ]
    create_cluster_main(english_xticks=english_labels, saved_filename='shanghai_districts_dendrogram.png')