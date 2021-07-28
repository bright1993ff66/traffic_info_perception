import pandas as pd
import os

from visualizations import hotspot_day_plot, generate_wordcloud, generate_wordcloud_in_given_month
from content_analysis.extract_keywords import extract_keywords_in_a_month
from data_paths import hotspot_figures, hotspot_text_path


def characterize_combined_hotspot(dataframe: pd.DataFrame, traffic_type: str, select_hotspot_ids: list):
    """
    Characterize the combined hotspot area
    :param dataframe: dataframe saving the Weibos spatially joined the hotspots
    :param traffic_type: the traffic type, either acc or cgs
    :param select_hotspot_ids: a list containing the hotspot ids
    :return: None. The characterization results are saved to local directory
    """

    assert 'hotspot_id' in dataframe, "The dataframe should contain the hotspot id information"
    assert traffic_type in ['acc', 'cgs'], "The traffic information type should be either acc or cgs"

    select_dataframe = dataframe.loc[dataframe['hotspot_id'].isin(select_hotspot_ids)]
    studied_months = [6, 7, 8]
    day_plot_path = os.path.join(hotspot_figures, 'day_plot')
    wordcloud_path = os.path.join(hotspot_figures, 'wordcloud')
    keyword_path = os.path.join(hotspot_figures, 'keywords')
    if traffic_type == 'acc':
        hotspot_day_plot(hotspot_dataframe=select_dataframe,
                         title='Accident-related Weibos in Priority Accident Hotspot',
                         save_filename='traffic_acc_hotspot_priority_per_day.png', set_percentile=97.5,
                         color_pos='#4BB8FA', color_neutral='#FAF838', color_neg='#FA6466', plot_threshold=False,
                         save_path=day_plot_path)
        print('For priority accident hotspots...')
        for month in studied_months:
            generate_wordcloud_in_given_month(dataframe=select_dataframe, month=month,
                                              save_path=wordcloud_path,
                                              save_filename='acc_priority_hotspot_month_{}'.format(month))
            keywords = extract_keywords_in_a_month(hotspot_dataframe=select_dataframe, month=month,
                                        top_N_keywords_considered=100,
                                        save_path=keyword_path,
                                        save_filename='acc_priority_hotspot_month_{}.csv'.format(month))
            print('Month: {}: {}'.format(month, keywords))

    else:
        hotspot_day_plot(hotspot_dataframe=select_dataframe,
                         title='Congestion-related Weibos in Priority Congestion Hotspot',
                         save_filename='traffic_cgs_hotspot_priority_per_day.png', set_percentile=97.5,
                         color_pos='#4BB8FA', color_neutral='#FAF838', color_neg='#FA6466', plot_threshold=False,
                         save_path=day_plot_path)
        print('For priority congestion hotspots...')
        for month in studied_months:
            generate_wordcloud_in_given_month(dataframe=select_dataframe, month=month,
                                              save_filename='cgs_priority_hotspot_month_{}'.format(month),
                                              save_path=wordcloud_path)
            keywords = extract_keywords_in_a_month(hotspot_dataframe=select_dataframe, month=month,
                                        top_N_keywords_considered=100,
                                        save_filename='cgs_priority_hotspot_month_{}.csv'.format(month),
                                        save_path=keyword_path)
            print('Month: {}: {}'.format(month, keywords))


def characterize_each_hotspot(dataframe, traffic_type):
    """
    Conduct the hotspot characterization for each hotspot
    :param dataframe: a pandas dataframe saving the Weibos posted in the hotspot
    :param traffic_type: the considered traffic type, either 'acc' or 'cgs'
    :return: None. The day plot, wordcloud and keywords are saved to local directory
    """
    assert 'hotspot_id' in dataframe, "The dataframe should contain the hotspot id information"
    assert traffic_type in ['acc', 'cgs'], "The traffic information type should be either acc or cgs"
    studied_months = [6, 7, 8]
    day_plot_path = os.path.join(hotspot_figures, 'day_plot')
    wordcloud_path = os.path.join(hotspot_figures, 'wordcloud')
    keyword_path = os.path.join(hotspot_figures, 'keywords')
    for hotspot_id, hotspot_dataframe in dataframe.groupby('hotspot_id'):
        print('Characterizing the hotspot {} for the traffic type {}...'.format(
            hotspot_id, traffic_type))
        if traffic_type == 'acc':
            hotspot_day_plot(hotspot_dataframe=hotspot_dataframe,
                             title='Accident-related Weibos in Accident Hotspot {} by Day'.format(hotspot_id),
                             save_filename='traffic_acc_hotspot_{}_per_day.png'.format(hotspot_id), set_percentile=97.5,
                             color_pos='#4BB8FA', color_neutral='#FAF838', color_neg='#FA6466', plot_threshold=False,
                             save_path=day_plot_path)
            for month in studied_months:
                generate_wordcloud_in_given_month(dataframe=hotspot_dataframe, month=month,
                                                  save_path=wordcloud_path,
                                                  save_filename='acc_hotspot_{}_month_{}'.format(hotspot_id, month))
                extract_keywords_in_a_month(hotspot_dataframe=hotspot_dataframe, month=month,
                                            top_N_keywords_considered=100,
                                            save_path=keyword_path,
                                            save_filename='acc_{}_month_{}.xlsx'.format(hotspot_id, month))
        else:
            hotspot_day_plot(hotspot_dataframe=hotspot_dataframe,
                             title='Congestion-related Weibos in Congestion Hotspot {} by Day'.format(hotspot_id),
                             save_filename='traffic_cgs_hotspot_{}_per_day.png'.format(hotspot_id), set_percentile=97.5,
                             color_pos='#4BB8FA', color_neutral='#FAF838', color_neg='#FA6466', plot_threshold=False,
                             save_path=day_plot_path)
            for month in studied_months:
                generate_wordcloud_in_given_month(dataframe=hotspot_dataframe, month=month,
                                                  save_filename='cgs_hotspot_{}_month_{}'.format(hotspot_id, month),
                                                  save_path=wordcloud_path)
                extract_keywords_in_a_month(hotspot_dataframe=hotspot_dataframe, month=month,
                                            top_N_keywords_considered=100,
                                            save_filename='cgs_{}_month_{}.xlsx'.format(hotspot_id, month),
                                            save_path=keyword_path)


if __name__ == '__main__':
    weibo_accident = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_acc_hotspot_join.txt'),
                                 index_col=0, encoding='utf-8')
    weibo_congestion = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_cgs_hotspot_join.txt'),
                                   index_col=0, encoding='utf-8')
    # Get the Weibos posted in the hotspots
    accident_hotspot = weibo_accident.loc[weibo_accident['hotspot_id'] != 0]
    conges_hotspot = weibo_congestion.loc[weibo_congestion['hotspot_id'] != 0]

    # Characterize each hotspot
    characterize_each_hotspot(dataframe=accident_hotspot, traffic_type='acc')
    characterize_each_hotspot(dataframe=conges_hotspot, traffic_type='cgs')

    # Characterize the priority hotspots
    characterize_combined_hotspot(dataframe=accident_hotspot, traffic_type='acc',
                                  select_hotspot_ids=[4, 7, 8, 11, 13])
    characterize_combined_hotspot(dataframe=conges_hotspot, traffic_type='cgs',
                                  select_hotspot_ids=[3, 4, 8, 12])