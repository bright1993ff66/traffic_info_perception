import numpy as np
import pandas as pd
from collections import Counter
import time
import os
import gensim.models as gs
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

import data_paths
from content_analysis.traffic_weibo import traffic_word_set_update
from process_text.text_preprocessing import preprocessing_weibo, create_stopwords_list
from utils import merge_dict

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='gensim')


class WeiboClassifierDataCreationWord2vec():

    def __init__(self, weibo_filename, word2vec_saving_path, classdataset_name, word2vec_model, weibo_vec_filename):
        """
        Initialize the data for training, validation, and testing; pretrained word vectors;
        :param weibo_filename: the name of the file to create the training, validation, and test data
        :param word2vec_saving_path: the path used to save the pretrained word2vec model
        :param classdataset_name: the name used to save the created sentence representation
        :param word2vec_model: the pretrained word2vec model
        :param weibo_vec_filename:  the name of the pretrained word2vec model
        """
        self.classdataset = classdataset_name
        # # Create the dataframe for training, validation, and test
        # # Only used when you have not created the training, validation, and test data
        # if 'xlsx' in weibo_filename:
        #     weibo_dataframe = pd.read_excel(os.path.join(data_paths.weibo_data_path, weibo_filename))
        # else:
        #     weibo_dataframe = pd.read_csv(os.path.join(data_paths.weibo_data_path, weibo_filename), encoding='utf-8',
        #                                   index_col=0)
        # weibo_label_2 = weibo_dataframe.loc[weibo_dataframe['label_3'] == 2]
        # weibo_label_1 = weibo_dataframe.loc[weibo_dataframe['label_3'] == 1]
        # weibo_label_0 = weibo_dataframe.loc[weibo_dataframe['label_3'] == 0].sample(
        #     6000 - weibo_label_1.shape[0] - weibo_label_2.shape[0])
        # weibo_select_concat = shuffle(pd.concat([weibo_label_0, weibo_label_1, weibo_label_2], axis=0))
        # self.considered_count = weibo_select_concat.shape[0]
        # weibo_select_reindex = weibo_select_concat.reset_index(drop=True)
        # print('Save the whole train, validation and test data to local...')
        # weibo_select_reindex.to_csv(os.path.join(data_paths.weibo_data_path, 'data_train_valid_test.csv'),
        #                             encoding='utf-8')
        # # Create the data for training, validation and test
        # train_valid_dataframe, test_dataframe = train_test_split(weibo_select_reindex, test_size=0.2)
        # train_dataframe, valid_dataframe = train_test_split(train_valid_dataframe, test_size=0.25)
        # train_dataframe.to_csv(os.path.join(data_paths.weibo_data_path, 'data_train.csv'), encoding='utf-8')
        # valid_dataframe.to_csv(os.path.join(data_paths.weibo_data_path, 'data_valid.csv'), encoding='utf-8')
        # test_dataframe.to_csv(os.path.join(data_paths.weibo_data_path, 'data_test.csv'), encoding='utf-8')
        train_dataframe = pd.read_csv(os.path.join(data_paths.weibo_data_path, 'data_train.csv'), index_col=0,
                                      encoding='utf-8')
        valid_dataframe = pd.read_csv(os.path.join(data_paths.weibo_data_path, 'data_valid.csv'), index_col=0,
                                      encoding='utf-8')
        test_dataframe = pd.read_csv(os.path.join(data_paths.weibo_data_path, 'data_test.csv'), index_col=0,
                                      encoding='utf-8')
        self.considered_count = train_dataframe.shape[0] + valid_dataframe.shape[0] + test_dataframe.shape[0]
        print('In total, we have {} weibos: {} for training, {} for validation, {} for testing'.format(
            self.considered_count, train_dataframe.shape[0], valid_dataframe.shape[0], test_dataframe.shape[0]))
        train_weibo_list = list(train_dataframe['text'])
        valid_weibo_list = list(valid_dataframe['text'])
        test_weibo_list = list(test_dataframe['text'])
        cleaned_train_weibo_list = []
        cleaned_valid_weibo_list = []
        cleaned_test_weibo_list = []
        cleaned_word_count_list = [] # Save the number of words counted in the Weibo dataset
        for weibo_train in train_weibo_list:
            weibo_train_string = str(weibo_train)
            cleaned_weibo_from_train = preprocessing_weibo(weibo_train_string, return_word_list=False)
            word_count_from_train = len(preprocessing_weibo(weibo_train_string, return_word_list=True))
            cleaned_train_weibo_list.append(cleaned_weibo_from_train)
            cleaned_word_count_list.append(word_count_from_train)
        for weibo_valid in valid_weibo_list:
            weibo_valid_string = str(weibo_valid)
            cleaned_weibo_from_valid = preprocessing_weibo(weibo_valid_string, return_word_list=False)
            word_count_from_valid = len(preprocessing_weibo(weibo_valid_string, return_word_list=True))
            cleaned_valid_weibo_list.append(cleaned_weibo_from_valid)
            cleaned_word_count_list.append(word_count_from_valid)
        for weibo_test in test_weibo_list:
            weibo_test_string = str(weibo_test)
            cleaned_weibo_from_test = preprocessing_weibo(weibo_test_string, return_word_list=False)
            word_count_from_test = len(preprocessing_weibo(weibo_test_string, return_word_list=True))
            cleaned_test_weibo_list.append(cleaned_weibo_from_test)
            cleaned_word_count_list.append(word_count_from_test)
        print('All the weibos have been processed for this file! Start generating word vectors...')
        self.train_weibos = cleaned_train_weibo_list
        self.valid_weibos = cleaned_valid_weibo_list
        self.test_weibos = cleaned_test_weibo_list
        self.train_labels = np.array(list(train_dataframe['label_3']))
        self.valid_labels = np.array(list(valid_dataframe['label_3']))
        self.test_labels = np.array(list(test_dataframe['label_3']))
        self.weibos_count = cleaned_word_count_list
        print("The 90% percentile is: {}".format(np.percentile(self.weibos_count, 90)))
        # Change the file path
        self.Word2Vec_model = word2vec_model
        self.word2vec_saving_path = word2vec_saving_path
        self.weibo_vec_file = weibo_vec_filename

    def data_creation(self):
        """
        Pack the representation of Weibo and label and save to local directory
        :return: None. The Weibo representation and label information have saved to the local directory
        """
        stopwords_list = create_stopwords_list(stopword_path=data_paths.chinese_stopword_path)
        vectorizer = CountVectorizer(min_df=1, stop_words=stopwords_list, ngram_range=(1, 1), analyzer='word')
        analyze = vectorizer.build_analyzer()
        # Create (Lxd) word embedding matrix corresponding to each tweet. L: number of words in tweet,
        # d: word vector dimension
        d = self.Word2Vec_model.vector_size
        assert d == 300, 'The dimension of word vector should be 300' # Check the dimension of the word vector
        L = 50

        weibo_represet_train = np.zeros((len(self.train_weibos), L, d), dtype=np.float32)
        for i in range(len(self.train_weibos)):
            words_seq = analyze(self.train_weibos[i])
            index = 0
            print(words_seq)
            for word in words_seq:
                if index < L:
                    try:
                        weibo_represet_train[i, index, :] = self.Word2Vec_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break
        weibo_represet_valid = np.zeros((len(self.valid_weibos), L, d), dtype=np.float32)
        for i in range(len(self.valid_weibos)):
            words_seq = analyze(self.valid_weibos[i])
            index = 0
            for word in words_seq:
                if index < L:
                    try:
                        weibo_represet_valid[i, index, :] = self.Word2Vec_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break
        weibo_represet_test = np.zeros((len(self.test_weibos), L, d), dtype=np.float32)
        for i in range(len(self.test_weibos)):
            words_seq = analyze(self.test_weibos[i])
            index = 0
            for word in words_seq:
                if index < L:
                    try:
                        weibo_represet_test[i, index, :] = self.Word2Vec_model[word]
                        index += 1
                    except KeyError:
                        pass
                else:
                    break

        filename = os.path.join(self.word2vec_saving_path, '{}'.format(self.weibo_vec_file[:-4]) + '_' +
                                self.classdataset+ str(self.considered_count) + '.pickle')
        # Use pickle to save the Weibo representation and label information locally
        with open(filename, 'wb') as f:
            pickle.dump([weibo_represet_train, weibo_represet_valid, weibo_represet_test, self.train_labels,
                         self.valid_labels, self.test_labels], f)
        print("Weibo word2vec matrix has been created as the input layer")

    def generate_word_count_distribution(self, save_path, save_filename):
        """
        Create the cumulative distribution of number of weibos for a weibo dataset
        :param save_path: the path used to save the created figure
        :param save_filename: the name of the created figure
        :return: None: The word count distribution is saved to local directory
        """
        figure, axis = plt.subplots(1, 1, figsize=(10, 8))
        sns.ecdfplot(self.weibos_count, ax=axis, color='red')
        axis.axvline(x=50, color='blue')
        axis.set_xlabel('Word Count')
        figure.savefig(os.path.join(save_path, save_filename))


class WeiboRepresentGeneration(object):

    def __init__(self, dataframe, word2vec_model, classifier):
        """
        Generate the representation of Weibos in a Weibo dataframe
        :param dataframe: a pandas dataframe saving the collected Weibos
        :param word2vec_model: a pretrained word2vec model
        :param classifier: a trained traffic information detection module
        """
        self.weibo_dataframe = dataframe
        self.Word2Vec_model = word2vec_model
        self.classifier = classifier

        assert 'text' in self.weibo_dataframe, "The Weibo dataframe should contain a column named text"

    def generate_repre_predict(self, predict_repost=True):
        """
        Generate the Weibo representation and make prediction
        :param predict_repost: whether predict the repost or not
        :return: The following information are returned
        total_count: total number of traffic relevant Weibo with location information;
        result_weibo_list: prediction based on the Weibo text
        result_reposts_list: prediction based on the repost text, if setting predict_repost=True
        """
        stopwords_list = create_stopwords_list(stopword_path=data_paths.chinese_stopword_path)
        vectorizer = CountVectorizer(min_df=1, stop_words=stopwords_list, ngram_range=(1, 1), analyzer='word')
        analyze = vectorizer.build_analyzer()
        # Create (Lxd) word embedding matrix corresponding to each tweet.
        # L: number of words in tweet,
        # d: word vector dimension
        d = self.Word2Vec_model.vector_size
        assert d == 300, 'The dimension of word vector should be 300'  # Check the dimension of the word vector
        L = 50  # Above 90 percentile value of number of words in a weibo dataset
        result_weibo_list, result_reposts_list = [], []
        for _, row in self.weibo_dataframe.iterrows():
            weibo = str(row['text'])
            # Cope with the Weibo
            # Make sure the candidate traffic-relevant Weibo contains one keyword
            if any(traffic_word in weibo for traffic_word in traffic_word_set_update):
                cleaned_weibo = preprocessing_weibo(weibo, return_word_list=False)
                weibo_represet = np.zeros((L, d), dtype=np.float32)
                words_seq = analyze(cleaned_weibo)
                index = 0
                for word in words_seq:
                    if index < L:
                        try:
                            weibo_represet[index, :] = self.Word2Vec_model[word]
                            index += 1
                        except KeyError:
                            pass
                    else:
                        break
                weibo_represent_reshape = weibo_represet.reshape((1, weibo_represet.shape[0],
                                                                  weibo_represet.shape[1], 1))
                predictions = self.classifier.predict(weibo_represent_reshape)
                final_prediction = np.argmax(predictions, axis=1)[0]
                result_weibo_list.append(final_prediction)
            else:
                result_weibo_list.append(0)

            # Cope with the reposts
            if predict_repost:
                reposts = str(row['retweeters_text'])
                if any(traffic_word in reposts for traffic_word in traffic_word_set_update):
                    cleaned_reposts = preprocessing_weibo(reposts, return_word_list=False)
                    reposts_represet = np.zeros((L, d), dtype=np.float32)
                    words_seq_reposts = analyze(cleaned_reposts)
                    index_repost = 0
                    for word in words_seq_reposts:
                        if index_repost < L:
                            try:
                                reposts_represet[index_repost, :] = self.Word2Vec_model[word]
                                index_repost += 1
                            except KeyError:
                                pass
                        else:
                            break
                    reposts_represent_reshape = reposts_represet.reshape((1, reposts_represet.shape[0],
                                                                      reposts_represet.shape[1], 1))
                    predictions_reposts = self.classifier.predict(reposts_represent_reshape)
                    final_prediction_repost = np.argmax(predictions_reposts, axis=1)[0]
                    result_reposts_list.append(final_prediction_repost)
                else:
                    result_reposts_list.append(0)
            else:
                pass
        weibo_count_dict = Counter(result_weibo_list)
        if predict_repost:
            repost_count_dict = Counter(result_reposts_list)
            combined_dict = merge_dict(weibo_count_dict, repost_count_dict)
        else:
            combined_dict = weibo_count_dict
        # Get the total number of traffic relevant Weibos, including with & without location information
        total_count = combined_dict.setdefault(1, 0) + combined_dict.setdefault(2, 0)
        print('In toal, We have got {} traffic relevant weibos. {} of them have location info in text'.format(
            total_count, combined_dict.setdefault(2, 0)))
        return total_count, result_weibo_list, result_reposts_list


if __name__ == '__main__':

    starting_time = time.time()
    print('Load the word vectors pretrained in Weibo...')
    weibo_fasttext_model = gs.KeyedVectors.load_word2vec_format(os.path.join(data_paths.word_vec_path,
                                                                             'sgns.weibo.bigram-char.bz2'))
    print('Generating the word embedding for traffic event detection...')
    word2vec = WeiboClassifierDataCreationWord2vec(weibo_filename='data_for_label_10000.xlsx',
                                                   word2vec_saving_path=data_paths.word_vec_path,
                                                   classdataset_name='Weibo_final',
                                                   word2vec_model=weibo_fasttext_model,
                                                   weibo_vec_filename='sgns.weibo.bigram-char.bz2')
    word2vec.generate_word_count_distribution(data_paths.figures_path, 'word_count_dist.png')
    word2vec.data_creation()

