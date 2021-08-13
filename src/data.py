def preprocess_dataframe(df):
    """
    Ensures that the sample is compatible with that used in the replication of
    the paper, i.e.

    - drops accidents after 2003
    - drops columns not used in the replication
    - drops rows with missing values in them
    """
    df = df[[feat for feat in df.columns if not feat.startswith('imp')]]
    df = df[~df.isna().max(axis=1)]
    df['modelyr'] = df['modelyr'].astype(int)
    df = df[df['year'] <= 2003]
    df['car_age'] = df['year'] - df['modelyr']

    return df
