def get_column_names(dataframe):
    if (dataframe is None):
        return None
    return list(dataframe.columns.values)

def get_nb_of_dimensions(dataframe):
    if (dataframe is None):
        return None
    return dataframe.shape[1]

def get_nb_of_rows(dataframe):
    if (dataframe is None):
        return None
    return dataframe.shape[0]

def get_number_column_names(dataframe):
    if (dataframe is None):
        return None
    return get_column_names(dataframe.select_dtypes(include=['number']))

def get_object_column_names(dataframe):
    if (dataframe is None):
        return None
    return get_column_names(dataframe.select_dtypes(include=['object']))

def get_unique_values(dataframe, column_name):
    if (dataframe is None):
        return None
    return dataframe[column_name].unique()