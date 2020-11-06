import gensim.models as gs
import gensim.corpora as corpora
import os
import pandas as pd
import pprint

from data_paths import random_seed, weibo_data_path
from process_text.text_preprocessing import preprocessing_weibo


def traffic_topic_model(dataframe, studied_text_column):
    """
    Get the topics based on a column of text in a Weibo dataframe
    :param dataframe: a Weibo dataframe
    :param studied_text_column: the name of the text column saving the studied text
    :return: the trained lda model
    """
    text_for_topic_model = [preprocessing_weibo(raw_tweet=text, return_word_list=True) for text in list(
        dataframe[studied_text_column])]
    id2word = corpora.Dictionary(text_for_topic_model)
    corpus = [id2word.doc2bow(text) for text in text_for_topic_model]
    lda_model = gs.ldamodel.LdaModel(corpus=corpus, id2word=id2word,
                                     num_topics=10, random_state=random_seed, update_every=1,
                                     chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    return lda_model


if __name__ == '__main__':
    weibo_data = pd.read_csv(os.path.join(weibo_data_path, 'combined_traffic_weibo_shanghai.csv'), encoding='utf-8',
                             index_col=0)
    sampled_weibo_data = weibo_data.sample(50)
    sampled_lda = traffic_topic_model(dataframe=sampled_weibo_data, studied_text_column='text')
    print(sampled_lda.print_topics())


