import pandas as pd
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

def get_dataset():
    folder_path = "data/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv"
    return pd.read_csv(folder_path, header=0)

def clean_data(df, categorie):
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    group = df.groupby('date', as_index=False)
    weighted_price = group[categorie].mean()
    return weighted_price

def split_data(data, train_percentage):
    length_data = int(len(data))
    train_size = int(length_data * train_percentage)
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set, test_set

def create_labels(dataset, sequence_length):
    sequence_length += 1
    seq_dataset = []
    for i in range(len(dataset) - sequence_length):
        seq_dataset.append(dataset[i: i + sequence_length])

    seq_dataset = np.array(seq_dataset)

    data_x = seq_dataset[:, :-1]
    data_y = seq_dataset[:, -1]

    return data_x, data_y

    # dataX, dataY = [], []
    # for i in range(len(data) - look_back - 1):
    #     a = data[i:(i + look_back), 0]
    #     dataX.append(a)
    #     dataY.append(data[i + look_back, 0])
    # return np.array(dataX), np.array(dataY)

    # length_data = int(len(data))
    # seq_dataset = []
    # for i in range(length_data - look_back):
    #     seq_dataset.append(data[i: i + look_back])
    # seq_dataset = np.array(seq_dataset)
    # data_x = seq_dataset[:, :-1]
    # data_y = seq_dataset[:, -1]
    # return data_x, data_y

def plot_original_data(data):
    plt.figure(figsize=(10, 10))
    plt.plot(data.date, data.Weighted_Price, lw=1, label='Original Price')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

def plot_training_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def build_model():
    model = Sequential()
    model.add(LSTM( input_shape=(None, 1),
                    units=50,
                    return_sequences=True))
    model.add(Dropout(0.35))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.35))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()
    return model

def train_model(model, x_train, y_train):
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.lstm.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    history = model.fit(x_train, y_train, batch_size=64,
                                epochs=5,
                                verbose=1,
                                validation_split=0.2,
                                callbacks=[checkpointer])
    return history

def plot_predicted_data(test_predict, train_predict, scaler, look_back, weighted_price):

    train_predict_unscaled = scaler.inverse_transform(train_predict)
    test_predict_unscaled = scaler.inverse_transform(test_predict)

    testPredictPlot = np.empty_like(weighted_price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict_unscaled) + (look_back * 2) + 1:len(weighted_price) - 1, :] = test_predict_unscaled

    # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
    trainPredictPlot = np.empty_like(weighted_price)
    # trainPredictPlot[:, :] = np.nan
    print(trainPredictPlot)
    trainPredictPlot[look_back:len(train_predict_unscaled) + look_back, :] = train_predict_unscaled

    # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
    testPredictPlot = np.empty_like(weighted_price)
    # testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict_unscaled) + (look_back * 2) + 1:len(weighted_price) - 1, :] = test_predict_unscaled

    plt.figure(figsize=(10, 10))
    plt.plot(weighted_price, 'g', label='original dataset')
    plt.plot(train_predict_unscaled, 'r', label='training set')
    plt.plot(testPredictPlot, 'b', label='predicted price/test set')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Price')

    plt.savefig('test2.png')
    plt.show()

