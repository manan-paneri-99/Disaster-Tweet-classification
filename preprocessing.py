import pandas as pd
from sklearn.model_selection import train_test_split


def fetch_data_as_df(path='./Data'):
    """
    :param path: path to data dump
    :return: data returned as dataframe
    """
    df = pd.read_csv(path + '/train.csv', low_memory=False)
    return df


def visualize_data(path='./Data'):
    """
    display data distribution
    :param path: path to data dump
    """
    df = fetch_data_as_df(path)
    df['target'].plot(kind='hist', title='Target dist')


def get_batches(train_size=0.8, path='./Data'):
    """
    get train and validation batches
    :param train_size: training set fraction
    :param path: path to data dump
    :return: training and validation dataframes
    """
    df = fetch_data_as_df(path)
    train_df, valid_df = train_test_split(df, random_state=43, train_size=train_size)
    return train_df, valid_df
