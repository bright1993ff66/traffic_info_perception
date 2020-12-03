import gensim.models as gs
import gensim.corpora as corpora
import os
import pandas as pd
import pprint

from data_paths import random_seed, weibo_data_path, hotspot_text_path
from process_text.text_preprocessing import preprocessing_weibo


def traffic_topic_model(dataframe, topic_num:int):
    """
    Get the topics based on a column of text in a Weibo dataframe
    :param dataframe: a Weibo dataframe
    :param topic_num: the number of topics considered in the topic model
    :return: the trained lda model
    """
    if 'retweeters_text' not in dataframe:
        dataframe_renamed = dataframe.rename(columns={'retweete_1': 'retweeters_text'})
    else:
        dataframe_renamed = dataframe.copy()
    text_list = list(dataframe_renamed['text'])
    retweet_list = list(dataframe_renamed['retweeters_text'])
    retweet_list_final = [text for text in retweet_list if text != 'no retweeters']
    combined_text_list = text_list + retweet_list_final
    combined_text_without_nan = list(filter(lambda x: str(x) != 'nan', combined_text_list))
    text_for_topic_model = [preprocessing_weibo(raw_tweet=text, return_word_list=True) for text in combined_text_without_nan]
    id2word = corpora.Dictionary(text_for_topic_model)
    corpus = [id2word.doc2bow(text) for text in text_for_topic_model]
    lda_model = gs.ldamodel.LdaModel(corpus=corpus, id2word=id2word,
                                     num_topics=topic_num, random_state=random_seed, update_every=1,
                                     chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    return lda_model


if __name__ == '__main__':

    acc_dataframe = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_in_acc_hotspots.csv'), encoding='utf-8',
                                   index_col=0)
    conges_dataframe = pd.read_csv(os.path.join(hotspot_text_path, 'weibo_in_conges_hotspots.csv'), encoding='utf-8',
                                 index_col=0)
    acc_lda = traffic_topic_model(acc_dataframe, topic_num=10)
    conges_lda = traffic_topic_model(conges_dataframe, topic_num=10)
    with open(os.path.join(weibo_data_path, 'acc_lda_topics.txt'), 'w', encoding='utf-8') as acc_file:
        acc_file.write("\n".join([result[1] for result in acc_lda.print_topics()]))
    with open(os.path.join(weibo_data_path, 'conges_lda_topics.txt'), 'w', encoding='utf-8') as conges_file:
        conges_file.write("\n".join([result[1] for result in conges_lda.print_topics()]))


