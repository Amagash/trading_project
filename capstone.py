import utilities as util
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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

    #plot original data
    plt.figure(figsize=(10, 10))
    plt.plot(data_scaled, 'g', label='original dataset')
    plt.legend(loc='upper left')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    # plt.show()

    train_set, test_set = util.split_data(data_scaled, train_percentage=0.85)

    x_train, y_train = util.create_labels(train_set, look_back=5)
    x_test, y_test = util.create_labels(test_set, look_back=5)

    model = util.build_model()


    #
    # history = model.fit(x_train, y_train,
    #                     batch_size=64,
    #                     epochs=30,
    #                     verbose=2,
    #                     validation_split=0.2)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()





if __name__ == "__main__":
    run()

