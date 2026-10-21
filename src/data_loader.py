import pandas as pd


def load_data(filepath):
    """
    Carrega um dataset CSV e retorna um DataFrame.
    """
    return pd.read_csv(filepath)