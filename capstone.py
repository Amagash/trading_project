import utilities
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



def run():
    #load data in dataframe
    data = utilities.get_dataset()
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

    utilities.split_data(data_scaled, train_percentage=0.85)


if __name__ == "__main__":
    run()