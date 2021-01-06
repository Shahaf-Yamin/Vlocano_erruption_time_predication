import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import glob
import plotly.express as px
from plotly.subplots import make_subplots
import os
from data import data_analyzer
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
from sklearn.preprocessing import StandardScaler
import pylab
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA


def plot_data(config,data):
    fig = px.histogram(
        data['train']['segments'],
        x="time_to_eruption",
        width=800,
        height=500,
        nbins=100,
        title='Time to eruption distribution'
    )
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    fig.write_image('outputs/Train-Time to eruption-Histogram.png')

    fig = px.line(
        data['train']['segments'],
        y="time_to_eruption",
        width=800,
        height=500,
        title='Time to eruption for all volcanos'
    )

    fig.write_image('outputs/Time to eruption for all volcanos-line.png')

def load_data(config):

    data = read_data(config)

    if config.with_train_histogram is True:
        plot_data(config, data)

    psd_pca_scaler = StandardScaler()
    psd_pca_transformer = decomposition.PCA(n_components=100)
    train_analyzer = data_analyzer.data_analyzer(data['train'], config, name='train',psd_scaler=psd_pca_scaler,pca_transformer=psd_pca_transformer)
    test_analyzer = data_analyzer.data_analyzer(data['test'], config, name='test',psd_scaler=psd_pca_scaler,pca_transformer=psd_pca_transformer)

    assert config.read_and_remove_redudant_features != config.with_feature_extraction or (config.with_feature_extraction is False and config.read_and_remove_redudant_features is False)

    '''
    Extract features
    '''
    if config.with_feature_extraction is True:
        train_analyzer.analyse_missing_observation()
        # # train_analyzer.calculate_statistics()
        train_analyzer.extract_data_features()

        test_analyzer.analyse_missing_observation()
        # test_analyzer.calculate_statistics()
        test_analyzer.extract_data_features()
        '''
        Remove redudant features
        '''
        X_train, y_train, drop_columns = train_analyzer.remove_redudant_features()
        X_test,test_segment_id = test_analyzer.remove_redudant_features(drop_columns)
    elif config.read_and_remove_redudant_features is True:
        train_analyzer.load_data_features_before_removing_features()
        test_analyzer.load_data_features_before_removing_features()
        X_train, y_train, drop_columns = train_analyzer.remove_redudant_features()
        X_test,test_segment_id = test_analyzer.remove_redudant_features(drop_columns)
    else:
        if config.with_redudant_features is True:
            print(f'Loading data before redudant feature remove procedure. version {config.feature_version} is loaded')
            X_train, y_train, drop_cols = train_analyzer.load_data_features_before_removing_features()
            X_test,test_segment_id = test_analyzer.load_data_features_before_removing_features()
        else:
            print(f'Loading data after redudant feature remove procedure. version {config.feature_version} is loaded')
            X_train, y_train, drop_cols = train_analyzer.load_data_features_after_removing_features()
            X_test,test_segment_id = test_analyzer.load_data_features_after_removing_features()
    for psd_index in range(50, config.max_psd_elements):
        try:
            X_train.drop(['psd_' + str(psd_index)], axis=1)
            X_test.drop(['psd_' + str(psd_index)], axis=1)
        except:
            print(f'{psd_index} already removed by correlation threshold')


    data = {'train':  (X_train, y_train), 'test': X_test}

    return data,test_segment_id

def preprocess_data(data,config):
    # Split into validation
    X_train, y_train = data['train']
    if config.with_validation:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=666, test_size=0.2, shuffle=True)
    else:
        X_val = None
        y_val = None

    # Prase the input data
    X_test = data['test']

    # Normalize
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if X_val is not None:
        X_val = scaler.transform(X_val)
        print(f"X_train_preprocess-size: {X_train.shape}, X_val_preprocess-size: {X_val.shape}, X_test_preprocess-size: {X_test.shape}")
    else:
        print(f"X_train_preprocess-size: {X_train.shape} X_test_preprocess-size: {X_test.shape}")

    print(f"shape of preprocess_model._mean: {scaler.mean_.shape}")

    # Use Kernel PCA in order to divide the the main contributed features
    model_ker_pca = KernelPCA(kernel='rbf', gamma=None, random_state=None)
    X_train = model_ker_pca.fit_transform(np.nan_to_num(X_train))
    X_test = model_ker_pca.transform(np.nan_to_num(X_test))

    if X_val is not None:
        X_val = model_ker_pca.transform(np.nan_to_num(X_val))
        print(f"X_train_reduced-size: {X_train.shape}, X_train_reduced-size:{X_val.shape}, X_test_reduced-size: {X_test.shape}")
    else:
        print(f"X_train_reduced-size: {X_train.shape}, X_test_reduced-size:{X_test.shape}")

    data = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': X_test}

    return data

def read_data(config):

    # Read fragments
    print('Reading fragments')
    train_frags = glob.glob("data/dataset/train/*")
    test_frags = glob.glob("data/dataset//test/*")
    print(f'Train: {len(train_frags)} number of fragments were founded')
    print(f'Test: {len(test_frags)} number of fragments were founded')
    print('Reading segments')
    train_segments = pd.read_csv("data/dataset/train.csv")
    sample_submission = pd.read_csv("data/dataset/sample_submission.csv")

    data = {'train': {'segments':train_segments,'frags': train_frags}, 'test': {'segments':sample_submission,'frags':test_frags}}

    return data

def transform(data, config):
    def transform_example(image,label):
        image, label = tf.cast(image, tf.float32), tf.cast(label, tf.int64)
        image = tf.divide(image, 255.)
        return image, label

    data['train'] = data['train'].map(transform_example)
    data['test'] = data['test'].map(transform_example)

    return data

def make_iterators(data, config):
    def augment_example(image, label):
        image = image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        return image,label

    train_iter = data['train'].map(augment_example).shuffle(1000).batch(config.batch_size, drop_remainder=True).take(-1)
    train_eval_iter = data['train'].batch(config.batch_size_eval).take(-1)
    test_iter = data['test'].batch(config.batch_size_eval).take(-1)

    iterators = {'train': train_iter,
                 'train_eval': train_eval_iter,
                 'test': test_iter}
    return iterators

def pca_decomposition(X, X_test,classifier_list=None):
    # x_mean = np.mean(X.values, axis=0)  # calculate the mean
    # X = X.values - x_mean  # normalize the signal
    NUMBER_OF_COMPENETS = 100
    #pca = decomposition.KernelPCA(n_components=NUMBER_OF_COMPENETS, kernel='rbf', gamma=None)
    pca = decomposition.PCA(n_components=NUMBER_OF_COMPENETS)
    pca.fit(X)
    X_after_dimension_reducation = pca.transform(X)

    pylab.figure()
    pylab.scatter(X_after_dimension_reducation[:, 0], X_after_dimension_reducation[:, 1],c=classifier_list)
    pylab.show()

    pca.fit(X_test)
    X_test_after_dimension_reducation = pca.transform(X_test)

    # U, S, V = np.linalg.svd(X)
    # eigen_values = S ** 2
    # power = 0
    # power_array = np.zeros_like(eigen_values)
    # power_indication = False
    # reduction_index = 0
    # for eigen_ind in range(len(power_array)):
    #     power += eigen_values[eigen_ind] / np.sum(eigen_values)
    #     power_array[eigen_ind] = power
    #     if power > 0.98 and power_indication is False:
    #         reduction_index = eigen_ind
    #         power_indication = True
    # X_after_dimension_reducation = np.matmul(X, np.transpose(V[:reduction_index, :]))
    # X_test_after_dimension_reducation = np.matmul(X_test.values, np.transpose(V[:reduction_index, :]))

    # print(np.linalg.norm(X_true_PCA-X_after_dimension_reducation))
    # plt.figure()
    # plt.grid()
    # plt.plot(np.linspace(1, len(power_array), len(power_array)), power_array)
    # plt.show()
    #
    # plt.figure()
    # plt.grid()
    # plt.plot(np.linspace(1, len(power_array[:reduction_index]), len(power_array[:reduction_index])), power_array[:reduction_index])
    # plt.show()

    return X_after_dimension_reducation, X_test_after_dimension_reducation, NUMBER_OF_COMPENETS
