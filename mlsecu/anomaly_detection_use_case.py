import pandas as pd
import numpy as np
from mlsecu.data_exploration_utils import get_column_names, get_object_column_names
from mlsecu.data_preparation_utils import get_one_hot_encoded_dataframe, remove_nan_through_mean_imputation
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def get_list_of_attack_types(dataframe):
    if (dataframe is None):
        return None
    return get_object_column_names(dataframe)

def get_nb_of_attack_types(dataframe):
    if (dataframe is None):
        return None
    return len(get_list_of_attack_types(dataframe))

def get_list_of_if_outliers(dataframe, outlier_fraction):
    if (dataframe is None):
        return None
    dataframe = remove_nan_through_mean_imputation(get_one_hot_encoded_dataframe(dataframe))
    clf = IsolationForest(contamination=outlier_fraction, random_state=42).fit(dataframe)
    dataframe['outliers'] = clf.predict(dataframe)
    return dataframe[dataframe['outliers'] == -1].index.tolist()

def get_list_of_lof_outliers(dataframe, outlier_fraction):
    if (dataframe is None):
        return None
    dataframe = remove_nan_through_mean_imputation(get_one_hot_encoded_dataframe(dataframe))
    dataframe['outliers'] = LocalOutlierFactor(contamination=outlier_fraction).fit_predict(dataframe)
    return dataframe[dataframe['outliers'] == -1].index.tolist()

def get_list_of_parameters(dataframe):
    return get_column_names(dataframe)

def get_nb_of_if_outliers(dataframe, outlier_fraction):
    if (dataframe is None):
        return None
    clf = get_list_of_if_outliers(dataframe, outlier_fraction)
    return len(clf)

def get_nb_of_lof_outliers(dataframe, outlier_fraction):
    if (dataframe is None):
        return None
    clf = get_list_of_lof_outliers(dataframe, outlier_fraction)
    return len(clf)

def get_nb_of_occurrences(dataframe):
    if (dataframe is None):
        return None
    return dataframe.shape[0]

def get_nb_of_parameters(dataframe):
    if (dataframe is None):
        return None
    return len(get_list_of_parameters(dataframe))