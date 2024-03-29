import googlemaps
import pandas as pd
import numpy as np
import os
import time
import requests
from collections import Counter

from data_paths import google_api_key, weibo_data_path, shanghai_jun_aug_traffic_sent
from process_text.text_preprocessing import preprocessing_traffic_accounts, preprocessing_weibo


def check_in_shanghai(loc_string):
    """
    Based on a location string, check whether we should consider this location
    :param loc_string: a location string
    :return: True if the location string contains at least a district level location
    """
    decision1 = ('Shanghai' in loc_string)
    decision2 = ('Shanghai, China' != loc_string)
    if decision1 & decision2:
        return True
    else:
        return False


def baidumap_geocode(text, ak):
    """
    Use baidu map to geocode an address string
    :param text: a text string containing location information
    :param ak: baidu api key
    :return: latitude, longitude
    """
    base_url = 'http://api.map.baidu.com/geocoding/v3/?address='
    url = base_url + text + '&output=json&ak=' + ak
    try:
        data=requests.get(url).json()
        lon = float(data.get('result').get('location').get('lng'))
        lat = float(data.get('result').get('location').get('lat'))
        return lon, lat
    except:
        return 0, 0


def geocoding_string(gmap_client: googlemaps.Client, string: str) -> list:
    """
    Geocode a text string
    :param gmap_client: the Google map client
    :param string: a string waiting to be geocoded
    :return: the geocode result
    """
    geocode_result = googlemaps.client.geocode(client=gmap_client, address=string,
                                               components={'administrative_area_level_1': 'Shanghai Shi',
                                                           'country': 'CN'},
                                               bounds={'northeast': [31.867652, 121.974040],
                                                       'southwest': [30.688885, 120.854422]})
    return geocode_result


def geocoding_reverse(gmap_client, latitude: float, longitude: float) -> list:
    """
    Reverse geocode a text string
    :param gmap_client: the Google map client
    :param latitude: the latitude value
    :param longitude: the longitude value
    :return: a list of location information which is near the given latitude and longitude
    """
    reverse_result = googlemaps.client.reverse_geocode(client=gmap_client, latlng=(latitude, longitude))
    return reverse_result


def geocode_weibo_dataframe(gmap_client, weibo_dataframe: pd.DataFrame, text_column_name: str, saved_filename: str):
    """
    Geocode the Weibo dataframe. A numpy array would be saved to local containing the geocoded result
    :param gmap_client: a Google Map client
    :param weibo_dataframe: a studied Weibo dataframe
    :param text_column_name: the text column we consider for geocoding
    :param saved_filename: the filename of the saved location list
    """
    geocode_result = []
    counter = 0
    print('We need to process {} Weibos'.format(weibo_dataframe.shape[0]))
    for index, row in weibo_dataframe.iterrows():
        print('Geocoding the {}th row'.format(index))
        cleaned_chinese_text = preprocessing_weibo(row[text_column_name], return_word_list=False)
        # places = [ent for ent in nlp(cleaned_chinese_text).ents if ent.label_ in ['GPE', 'LOC', 'FAC', 'ORG']]
        geocode_one_weibo = geocoding_string(gmap_client=gmap_client, string=cleaned_chinese_text)
        geocode_result.append(geocode_one_weibo)
        counter += 1
        if counter % 50 == 0:
            print('Sleep for a while...')
            time.sleep(np.random.randint(low=20, high=40))
            print('Done! Start again!')
    print('Finished!')
    print('Saving the geocode result...')
    np.save(os.path.join(weibo_data_path, saved_filename), geocode_result)


def geocode_offical(gmap_client, official_traffic_data: pd.DataFrame,
                    save_filename: str = 'geocode_traffic_account_final.npy'):
    """
    Geocode the traffic-related Weibo posted by official account. A numpy array would be saved to local containing
    the geocoded result
    :param gmap_client: the Google map client
    :param official_traffic_data: the scrapped traffic Weibos posted by official traffic accounts
    :param save_filename: the name of the file saved to local
    """
    geocode_result = []
    counter = 0
    print('Geocoding the traffic weibo posted by Lexing Shanghai')
    if 'filtered_text' in official_traffic_data:
        text_column_name = 'filtered_text'
    else:
        text_column_name = 'text'
    for index, row in official_traffic_data.iterrows():
        print('Coping with the {}th weibo'.format(counter))
        cleaned_chinese_text = preprocessing_traffic_accounts(row[text_column_name])
        # places = [ent for ent in nlp(cleaned_chinese_text).ents if ent.label_ in ['GPE', 'LOC', 'FAC', 'ORG']]
        geocode_one_weibo = geocoding_string(gmap_client=gmap_client, string=cleaned_chinese_text)
        geocode_result.append(geocode_one_weibo)
        counter += 1
    print('Done!')
    print('Saving the geocode result...')
    np.save(os.path.join(weibo_data_path, save_filename), geocode_result)


def get_geocoded_not_geocoded_weibos(path: str):
    """
    Get the traffic weibos with latiude longitude info and traffic Weibo without latitude and longitude info
    :param path: the studied data path
    :return:
    """
    dataframe_list = []
    print('Get the geocoded and not geocoded traffic-related weibos...')
    for file in os.listdir(path):
        print('Coping with the file: {}'.format(file))
        dataframe = pd.read_csv(os.path.join(path, file), encoding='utf-8', index_col=0)
        dataframe_list.append(dataframe)
    concat_data = pd.concat(dataframe_list)
    geocoded_data = concat_data.loc[concat_data['lat'] != 'Not Given']
    not_geocoded_data = concat_data.loc[concat_data['lat'] == 'Not Given']
    geocoded_data_copy = geocoded_data.copy()
    not_geocoded_data_copy = not_geocoded_data.copy()
    # The index val of geocoded data is created to find Weibos after arcmap processing
    geocoded_data_copy['index_val'] = list(range(geocoded_data_copy.shape[0]))
    print('Save the geocoded and non-geocoded data to local...')
    geocoded_data_copy.to_csv(os.path.join(weibo_data_path, 'geocoded_traffic_weibos.csv'), encoding='utf-8')
    not_geocoded_data_copy.to_csv(os.path.join(weibo_data_path, 'non_geocoded_traffic_weibos.csv'), encoding='utf-8')
    return geocoded_data_copy, not_geocoded_data_copy


def process_nongeocoded_traffic(dataframe):
    """
    Process the nongeocoded Weibos. The following two types of nongeocoded Weibos are considered:
    - Select the Weibo data whose weibo is labelled as 2 but reposts are labelled as 0 or 1
    - Select the Weibo data whose reposts is labelled as 2
    So either the Weibo or the repost should be labeled as 2
    :param dataframe: the nongeocoded Weibo dataframe
    :return: weibo traffic data, repost traffic data
    """
    # Get the Weibos which are traffic relevant and have location information in the text
    weibo_traffic_data = dataframe[(dataframe['traffic_weibo'] == 2) & (dataframe['traffic_repost'].isin([0, 1]))]
    weibo_traffic_data_reindex = weibo_traffic_data.reset_index(drop=True)
    weibo_traffic_data_reindex['index_val'] = list(range(weibo_traffic_data_reindex.shape[0]))

    # Get the Weibos which repost Weibos that are traffic relevant and have location information
    repost_traffic_data = dataframe[dataframe['traffic_repost'] == 2]
    repost_traffic_data_reindex = repost_traffic_data.reset_index(drop=True)
    repost_traffic_data_reindex['index_val'] = list(range(repost_traffic_data_reindex.shape[0]))

    return weibo_traffic_data_reindex, repost_traffic_data_reindex


def construct_official_location_dataframe(location_list: list, text_list: list, time_list: list) -> pd.DataFrame:
    """
    Construct the location dataframe for geocoding
    :param location_list: the geocoded location list given by the Google Geocoding API
    :param text_list: the text content of Weibos posted by official traffic accounts
    :param time_list: the time of Weibos posted by official traffic accounts
    :return: a pandas dataframe where columns are ['location', 'lat', 'lon']
    """
    index_val_list = list(range(len(location_list)))
    result_dataframe = pd.DataFrame(columns=['time', 'location', 'loc_lat', 'loc_lon', 'index_val', 'text'])
    address_list = []
    latitude_list = []
    longitude_list = []
    result_index_list = []
    result_weibo_text_list = []
    result_weibo_time_list = []
    for index_val, locations, weibo_text, weibo_time in zip(index_val_list, location_list, text_list, time_list):
        for loc in locations:
            if not loc: continue
            if type(loc) == list:
                address_list.append(loc[0]['formatted_address'])
                latitude_list.append(loc[0]['geometry']['location']['lat'])
                longitude_list.append(loc[0]['geometry']['location']['lng'])
                result_index_list.append(index_val)
                result_weibo_text_list.append(weibo_text)
                result_weibo_time_list.append(weibo_time)
            elif type(loc) == dict:
                address_list.append(loc['formatted_address'])
                latitude_list.append(loc['geometry']['location']['lat'])
                longitude_list.append(loc['geometry']['location']['lng'])
                result_index_list.append(index_val)
                result_weibo_text_list.append(weibo_text)
                result_weibo_time_list.append(weibo_time)
            else:
                print('Something wrong with the loc: {}. Check!'.format(loc))
                break
    result_dataframe['location'] = address_list
    result_dataframe['loc_lat'] = latitude_list
    result_dataframe['loc_lon'] = longitude_list
    result_dataframe['index_val'] = result_index_list
    result_dataframe['text'] = result_weibo_text_list
    result_dataframe['time'] = result_weibo_time_list
    return result_dataframe


def construct_weibo_location_dataframe(location_list: list) -> pd.DataFrame:
    """
    Construct the location dataframe for geocoding
    :param location_list: the geocoded location list given by the Google Geocoding API
    :param text_list: the text content of Weibos posted by official traffic accounts
    :return: a pandas dataframe where columns are ['location', 'lat', 'lon']
    """
    index_val_list = list(range(len(location_list)))
    result_dataframe = pd.DataFrame(columns=['location', 'loc_lat', 'loc_lon', 'index_val', 'text'])
    address_list = []
    latitude_list = []
    longitude_list = []
    result_index_list = []
    result_weibo_text_list = []
    for index_val, locations, weibo_text in zip(index_val_list, location_list):
        for loc in locations:
            if not loc: continue
            if type(loc) == list:
                address_list.append(loc[0]['formatted_address'])
                latitude_list.append(loc[0]['geometry']['location']['lat'])
                longitude_list.append(loc[0]['geometry']['location']['lng'])
                result_index_list.append(index_val)
                result_weibo_text_list.append(weibo_text)
            elif type(loc) == dict:
                address_list.append(loc['formatted_address'])
                latitude_list.append(loc['geometry']['location']['lat'])
                longitude_list.append(loc['geometry']['location']['lng'])
                result_index_list.append(index_val)
                result_weibo_text_list.append(weibo_text)
            else:
                print('Something wrong with the loc: {}. Check!'.format(loc))
                break
    result_dataframe['location'] = address_list
    result_dataframe['loc_lat'] = latitude_list
    result_dataframe['loc_lon'] = longitude_list
    result_dataframe['index_val'] = result_index_list
    result_dataframe['text'] = result_weibo_text_list
    return result_dataframe


def merge_data_using_index(data_with_sent: pd.DataFrame, filtered_location_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the Weibo dataframe with sentiment and the filtered location dataframe
    :param data_with_sent: dataframe with sentiment label, weibo & repost text, etc
    :param filtered_location_dataframe: a dataframe with filtered locations (in shanghai)
    :return: a combined dataframe with location information, location lat & lon and weibo repost content
    """
    assert 'index_val' in data_with_sent, "The dataframe with sentiment should have index_val column"
    assert 'index_val' in filtered_location_dataframe, 'The location dataframe should have index_val column'
    combined_dataframe = pd.merge(data_with_sent, filtered_location_dataframe, on=['index_val'])
    return combined_dataframe


def create_nongeocoded_main_dataframe(studied_weibo_data:pd.DataFrame, studied_repost_data:pd.DataFrame):
    """
    Create the dataframe for two types of nongeocoded data
    :param studied_weibo_data: the studied weibo data of which the repost is not labelled as 2
    :param studied_repost_data: the studied weibo data of which the repost is labelled as 2
    :return: two pandas dataframe saving the weibo data for arcmap and repost data for arcmap
    """
    # Load the locations produced by the Google Geocoding API
    nongeocoded_weibo_locations = np.load(os.path.join(weibo_data_path, 'non_geocode_weibo_traffic_locations.npy'),
                                          allow_pickle=True)
    nongeocoded_repost_locations = np.load(os.path.join(weibo_data_path, 'non_geocode_repost_traffic_locations.npy'),
                                           allow_pickle=True)
    # Construct the location dataframe based on the geocoded list
    nongeocoded_weibo_loc_dataframe = construct_weibo_location_dataframe(nongeocoded_weibo_locations)
    nongeocoded_repost_loc_dataframe = construct_weibo_location_dataframe(nongeocoded_repost_locations)
    nongeocoded_weibo_loc_shanghai = nongeocoded_weibo_loc_dataframe[nongeocoded_weibo_loc_dataframe.apply(
        lambda row: check_in_shanghai(row['location']), axis=1)]
    nongeocoded_repost_loc_shanghai = nongeocoded_repost_loc_dataframe[nongeocoded_repost_loc_dataframe.apply(
        lambda row: check_in_shanghai(row['location']), axis=1)]
    # Create the main dataframe for the following analysis. Weibos with multiple locations would be shown in multiple
    # rows in the created dataframe
    weibo_main_data = merge_data_using_index(data_with_sent=studied_weibo_data,
                                             filtered_location_dataframe=nongeocoded_weibo_loc_shanghai)
    repost_main_data = merge_data_using_index(data_with_sent=studied_repost_data,
                                              filtered_location_dataframe=nongeocoded_repost_loc_shanghai)
    # Save the data to local, pass them to arcmap, and get the Weibos posted in Shanghai
    weibo_main_data.to_csv(os.path.join(weibo_data_path, 'nongeocoded_weibo_data_for_arcmap.csv'), encoding='utf-8')
    repost_main_data.to_csv(os.path.join(weibo_data_path, 'nongeocoded_repost_data_for_arcmap.csv'), encoding='utf-8')
    return weibo_main_data, repost_main_data
