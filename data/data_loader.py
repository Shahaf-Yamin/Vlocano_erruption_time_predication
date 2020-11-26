import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import glob
import plotly.express as px
from plotly.subplots import make_subplots
import os
from data import data_analyzer

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

    train_analyzer = data_analyzer.data_analyzer(data['train'], config, name='train')
    test_analyzer = data_analyzer.data_analyzer(data['test'], config, name='test')

    '''
    Extract features
    '''
    if config.with_feature_extraction is True:
        train_analyzer.analyse_missing_observation()
        train_analyzer.extract_data_features()

        test_analyzer.analyse_missing_observation()
        test_analyzer.extract_data_features()
        '''
        Remove redudant features
        '''
        X_train, y_train = train_analyzer.remove_redudant_features()
        X_test = test_analyzer.remove_redudant_features()
    else:
        if config.with_redudant_features is True:
            print(f'Loading data before redudant feature remove procedure. version {config.feature_version} is loaded')
            X_train, y_train = train_analyzer.load_data_features_before_removing_features()
            X_test = test_analyzer.load_data_features_before_removing_features()
        else:
            print(f'Loading data after redudant feature remove procedure. version {config.feature_version} is loaded')
            X_train, y_train = train_analyzer.load_data_features_after_removing_features()
            X_test = test_analyzer.load_data_features_after_removing_features()

    data = {'train':  (X_train, y_train), 'test': X_test}

    #transformed_data = transform(data, config)

    #data_iterators = make_iterators(transformed_data, config)

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
