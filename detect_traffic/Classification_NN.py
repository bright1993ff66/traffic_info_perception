# some necessary packages
import time
import pickle
import os
import numpy as np
import itertools
import pandas as pd
import gensim.models as gs

# import the built packages
import data_paths

# build models
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from detect_traffic.Compute_representation_for_each_tweet import WeiboRepresentGeneration

# model evaluation
from sklearn.metrics import confusion_matrix, f1_score

np.random.seed(data_paths.random_seed)


# Build the classification model
# Firstly, we create a baseline model based solely on tweet representation
def conv2d_lstm(dropout_rate):
    # Get the input information - tweet only
    weibo_input = Input(shape=(60, 300, 1), name='weibo_input') # row = 60; col = 300, channel = 1
    # Create the convolutional layer
    # Input: batch_shape + (rows=60, cols=300, channels=1)
    # Output: batch_shape + (new_rows, new_cols, filters)
    conv2d = Conv2D(filters=100, kernel_size=(2, 300), padding='valid',
                    activation='relu', use_bias=True, name='conv_1', data_format='channels_last')(weibo_input)
    reshaped_conv2d = Reshape((59, 100), name='reshape_1')(conv2d)
    conv_drop = Dropout(dropout_rate, name='dropout_1')(reshaped_conv2d)
    # Creat the LSTM layer
    # Input: (batch, timesteps, feature)
    # Output: (batch, units)
    lstm = LSTM(units=100, return_state=False, activation='tanh',
                recurrent_activation='hard_sigmoid', name='lstm_1')(conv_drop)
    lstm_drop = Dropout(dropout_rate, name='dropout_2')(lstm)
    dense_1 = Dense(100, activation='relu', name='dense_1')(lstm_drop)
    output = Dense(3, activation='softmax',
                   kernel_regularizer=regularizers.l2(0.01),
                   name='output_3_classes')(dense_1)
    # Build the model
    model = Model(inputs=weibo_input, outputs=output)
    return model


if __name__ == '__main__':

    start_time = time.time()

    print('Preparing the data and the label...')
    # Load the word vectors
    filename = os.path.join(data_paths.word_vec_path, 'sgns.weibo.bigram-char_Weibo.pickle')
    with open(filename, mode='rb') as f:
        Train_X, Valid_X, Test_X, Train_Y, Valid_Y, Test_Y = pickle.load(f, encoding='utf-8')
    Train_X_reshaped = Train_X.reshape((Train_X.shape[0], Train_X.shape[1], Train_X.shape[2], 1))
    Valid_X_reshaped = Valid_X.reshape((Valid_X.shape[0], Valid_X.shape[1], Valid_X.shape[2], 1))
    Test_X_reshaped = Test_X.reshape((Test_X.shape[0], Test_X.shape[1], Test_X.shape[2], 1))

    # Construct and compile the LSTM model
    class_num = len(set(Train_Y))
    Train_Y_categorical = keras.utils.to_categorical(Train_Y, num_classes=class_num)
    Valid_Y_categorical = keras.utils.to_categorical(Valid_Y, num_classes=class_num)
    Test_Y_categorical = keras.utils.to_categorical(Test_Y, num_classes=class_num)
    print('Data preprocessing is done!')
    print('Loading the hyperparameter settings...')
    # Specify the GridSearch parameters
    dropout_rates = [0.4, 0.5, 0.6]
    batch_sizes = [16, 32, 64]
    learning_rates = [0.0001, 0.0002, 0.001]
    grid_list = [dropout_rates, batch_sizes, learning_rates]
    hyperparameter_list = list(itertools.product(*grid_list))
    print(hyperparameter_list[:8])
    print('The number of parameter setting is: {}'.format(len(hyperparameter_list)))

    print('Building the model...')
    # Output model summary and compile the model

    for hyperparameters in hyperparameter_list:
        print('Coping with the hyperparameter setting: dropout rate: {}; batch_size: {}; learning_rate: {}'.format(
            hyperparameters[0], hyperparameters[1], hyperparameters[2]))
        dropout_rate, batch_size, learning_rate = hyperparameters[0], hyperparameters[1], hyperparameters[2]
        model = conv2d_lstm(dropout_rate=dropout_rate)
        model.summary()
        # # plot_model(model, 'cnn_lstm_model.png')
        # optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        # mc = ModelCheckpoint(os.path.join(data_paths.detect_traffic_path, 'best_models',
        #                                   'best_model_{}_{}_{}.h5'.format(dropout_rate, batch_size, learning_rate)),
        #                      monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy',
        #                        metrics=[categorical_accuracy, categorical_crossentropy])
        # # Train the model
        # model.fit(x=Train_X_reshaped, y=Train_Y_categorical, epochs=50, batch_size=batch_size, shuffle=True,
        #           validation_data=[Valid_X_reshaped, Valid_Y_categorical], verbose=0, callbacks=[es, mc])
        #
        # # Evaluate the model on test data, return a list containing the loss and the specified metrics
        # # In this case, our metrics are categorical accuracy and categorical cross entropy
        # model.evaluate(x=Test_X_reshaped, y=Test_Y_categorical, batch_size=64, verbose=1)
        #
        # predictions = np.argmax(model.predict(Test_X_reshaped), axis=1)
        # print('The performance on the test data is: {}'.format(confusion_matrix(Test_Y, predictions)))
        # print('The F1 score is: {}'.format(f1_score(y_pred=predictions, y_true=Test_Y, average='macro')))

    # # Load the best model and make prediction
    # weibo_fasttext_model = gs.KeyedVectors.load_word2vec_format(os.path.join(data_paths.word_vec_path,
    #                                                                          'sgns.weibo.bigram-char.bz2'))
    # model = keras.models.load_model(os.path.join(data_paths.detect_traffic_path, 'best_models',
    #                                              'best_model_{}_{}_{}.h5'.format(0.6, 32, 0.001)))
    # # Load a dataframe and make prediction
    # for file in os.listdir(data_paths.shanghai_jun_aug):
    #     print('Making the prediction for file: {}'.format(file))
    #     dataframe = pd.read_csv(os.path.join(data_paths.shanghai_jun_aug, file), encoding='utf-8', index_col=0)
    #
    #     generate_repre_obj = WeiboRepresentGeneration(dataframe=dataframe, word2vec_model=weibo_fasttext_model,
    #                                                   classifier=model)
    #     results_weibos, results_reposts = generate_repre_obj.generate_repre_predict()
    #     dataframe_copy = dataframe.copy()
    #     dataframe_copy['traffic_weibo'] = results_weibos
    #     dataframe_copy['traffic_repost'] = results_reposts
    #     dataframe_copy.to_csv(os.path.join(data_paths.shanghai_jun_aug_traffic, file[:-4]+'_traffic.csv'),
    #                           encoding='utf-8')

