import pandas as pd

def get_dataset():
    folder_path = "data/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv"
    return pd.read_csv(folder_path, header=0)

def split_data(data, train_percentage, look_back):
    length_data = int(len(data))
    print(length_data)
    train_size = int(length_data * train_percentage)
    train_set = data[:train_size]
    test_set = data[train_size:]
    print(len(train_set))
    print(len(test_set))
    print(len(train_set)+len(test_set))

