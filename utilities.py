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

def split_data(data, train_percentage):
    length_data = int(len(data))
    train_size = int(length_data * train_percentage)
    train_set = data[:train_size]
    test_set = data[train_size:]
    return train_set, test_set

def create_labels(data, look_back):
    look_back+=1
    length_data = int(len(data))
    seq_dataset = []
    for i in range(length_data - look_back):
        seq_dataset.append(data[i: i + look_back])

    seq_dataset = np.array(seq_dataset)

    data_x = seq_dataset[:, :-1]
    data_y = seq_dataset[:, -1]

    return data_x, data_y

def plot_original_data(data):
    #plot original data
    plt.figure(figsize=(10, 10))
    plt.plot(data, 'g', label='original dataset')
    plt.legend(loc='upper left')
    plt.xlabel('Timestamp')
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



