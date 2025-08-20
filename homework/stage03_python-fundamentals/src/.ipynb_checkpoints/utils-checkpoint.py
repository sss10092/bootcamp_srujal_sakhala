def get_summary_stats(df):
    """
    Returns summary statistics for all numeric columns.
    """
    return df.describe()
