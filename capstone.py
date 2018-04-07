import utilities as util
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def run():
    #load data in dataframe
    data = util.get_dataset()
    # print(data.head())
    # print(data.tail())

    weighted_price = data.Weighted_Price.values.astype('float32')
    # print(weighted_price)
    weighted_price = weighted_price.reshape(len(weighted_price), 1)
    # print(weighted_price)

    #scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(weighted_price)
    # print(data_scaled)

    look_back = 5
    train_set, test_set = util.split_data(data_scaled, train_percentage=0.85)
    x_train, y_train = util.create_labels(train_set, look_back=5)
    x_test, y_test = util.create_labels(test_set, look_back=5)

    model = util.build_model()
    history = util.train_model(model, x_train, y_train)
    util.plot_training_history(history)
    model.load_weights('saved_models/weights.best.lstm.hdf5')



if __name__ == "__main__":
    run()

