import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fill_missing_median(df, columns=None):
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(include="number").columns
    for col in columns:
        median_val = df_copy[col].median()
        df_copy[col] = df_copy[col].fillna(median_val)
    return df_copy

def drop_missing(df, threshold=0.5):
    df_copy = df.copy()
    missing_fraction = df_copy.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index
    df_copy.drop(columns=cols_to_drop, inplace=True)
    return df_copy

def normalize_data(df, columns=None):
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.select_dtypes(include="number").columns

    scaler = MinMaxScaler()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy