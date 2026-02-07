import pandas as pd

def load_data(file_path):
    """Load cyber threat dataset"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Basic cleaning step"""
    return df.dropna()
