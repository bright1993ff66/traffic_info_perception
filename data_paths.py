import os

api_key = r'AIzaSyBrzuittddSECsBFs4-uKMY5p94VrAiubg'
random_seed = 7

# For local computer
project_path = r'D:\Projects\Traffic_info_perception'
desktop = r'C:\Users\Public\Desktop'
data_path = os.path.join(project_path, 'data')
weibo_data_path = os.path.join(data_path, 'weibo')
chinese_stopword_path = os.path.join(data_path, 'stopword')
word_vec_path = os.path.join(project_path, 'word_vectors')
detect_traffic_path = os.path.join(project_path, 'detect_traffic')
figures_path = os.path.join(project_path, 'figures')
shapefile_path = os.path.join(project_path, 'shapefiles')
other_path = os.path.join(project_path, 'other')

# For hard drive
hard_drive = r'F:\sina_weibo\Weibo-Shanghai-Data'
shanghai_apr_may = os.path.join(hard_drive, 'shanghai_2012_apr_may')
shanghai_jun_aug = os.path.join(hard_drive, 'shanghai_2012_jun_aug')
shanghai_jun_aug_prediction = os.path.join(hard_drive, 'shanghai_2012_jun_aug_predictions')
shanghai_jun_aug_traffic = os.path.join(hard_drive, 'shanghai_2012_jun_aug_traffic')
shanghai_jun_aug_traffic_sent = os.path.join(hard_drive, 'shanghai_2012_jun_aug_traffic_sent')