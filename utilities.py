import pandas as pd

def get_dataset():
    folder_path = "data/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv"
    return pd.read_csv(folder_path, header=0)