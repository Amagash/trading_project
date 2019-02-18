# Machine Learning Engineer Nanodegree
## Capstone Proposal
Tiffany Souterre  
April 10th, 2018

## Proposal
Propose a Recurent Neural Network model to forecast market price of Bitcoin.

### Domain Background

Within a decade, cryptocurrencies have become a global phenomenon and is changing the way transactions are processed. It all started with the Bitcoin, released by Satoshi Nakamoto in 2009. He invented a way to build a decentralized digital cash system.

>*Announcing the first release of Bitcoin, a new electronic cash system that uses a peer-to-peer network to prevent double-spending. It’s completely decentralized with no server or central authority.* 
– Satoshi Nakamoto, 09 January 2009, announcing Bitcoin on SourceForge.

Before cryptocurrencies, double spending was prevented by central servers keeping records of balances. With cryptocurrencies there is no central server, therefore every peer in the network need to have a list with all transactions to check if future transactions are valid or an attempt to double spend. With Bitcoins, transaction blocks contain a SHA-256 cryptographic hash of previous transaction blocks, and are thus "chained" together. This blockchain serves as an immutable public ledger of all transactions that have ever occurred. Satoshi achieved consensus without a central authority<sup>[1]</sup>. Cryptocurrencies allow for transparent transactions without the need of a bank or any institution in between. They have become more popular in the past year and might become one of the principal way of doing transaction in the futur.

Although stock price prediction has been around for quite a while, making accurate prediction for cryptocurrencies stock price - considered very volatile - represents a real challenge for data scientists.
With the advent of machine learning, new techniques are being used such as Recurrent Neural Networks (RNN)<sup>[2]</sup>. In this project, I wish to explore Bitcoin data and try to apply my own neural network to take up the challenge of Bitcoin stock price prediction.


### Problem Statement
In 2017, it was estimated that there were 2.9 to 5.8 million unique users using a cryptocurrency wallet <sup>[3]</sup>. On 3 March 2017, the price of a bitcoin surpassed the market value of an ounce of gold for the first time at $1,268 and it reached its highest value on 15 December 2017 at $17,900 <sup>[4]</sup> although is has rapidly decreased since then.

The price of bitcoins has gone through various cycles of appreciation and depreciation. The behavior of bitcoin value is considered very volatile and not very well understood. This project aims at leveraging machine learning algorithms to generate financial models to better understand bitcoin behavior.

The goal is to be able to forecast the market price of bitcoin for the next n days given the history of market price variation.


### Datasets and Inputs

A dataset containing bitcoin exchanges for the time period of April 28, 2013 to February 20, 2018 can be found on Kaggle <sup>[5]</sup>. I will more precisely focus on the bitcoin_cash_price.csv containing the following features: 

- Date : Date of observation
- btc_market_price : Average USD market price across major bitcoin exchanges.
- btc_total_bitcoins : The total number of bitcoins that have already been mined.
- btc_market_cap : The total USD value of bitcoin supply in circulation.
- btc_trade_volume : The total USD value of trading volume on major bitcoin exchanges.
- btc_blocks_size : The total size of all block headers and transactions.
- btc_avg_block_size : The average block size in MB.
- btc_n_orphaned_blocks : The total number of blocks mined but ultimately not attached - to the main Bitcoin blockchain.
- btc_n_transactions_per_block : The average number of transactions per block.
- btc_median_confirmation_time : The median time for a transaction to be accepted into a mined block.
- btc_hash_rate : The estimated number of tera hashes per second the Bitcoin network is performing.
- btc_difficulty : A relative measure of how difficult it is to find a new block.
- btc_miners_revenue : Total value of coinbase block rewards and transaction fees paid to miners.
- btc_transaction_fees : The total value of all transaction fees paid to miners.
- btc_cost_per_transaction_percent : miners revenue as percentage of the transaction volume.
- btc_cost_per_transaction : miners revenue divided by the number of transactions.
- btc_n_unique_addresses : The total number of unique addresses used on the Bitcoin blockchain.
- btc_n_transactions : The number of daily confirmed Bitcoin transactions.
- btc_n_transactions_total : Total number of transactions.
- btc_n_transactions_excluding_popular : The total number of Bitcoin transactions, excluding the 100 most popular addresses.
- btc_n_transactions_excluding_chains_longer_than_100 : The total number of Bitcoin transactions per day excluding long transaction chains.
- btc_output_volume : The total value of all transaction outputs per day.
- btc_estimated_transaction_volume : The total estimated value of transactions on the Bitcoin blockchain.
- btc_estimated_transaction_volume_usd : The estimated transaction value in USD value.

### Solution Statement

Long short-term memory (LSTM) networks appear to be an ideal model candidate to solve the problem of forecasting market prices.
LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. 


### Benchmark Model

Other models exist for studying time series. I will compare the LSTM model to a classic Recurent Neural Network (RNN) and a Multilayer Perceptron (MLP). A MLP is a class of feedforward artificial neural network


### Evaluation Metrics

The evalutation metric that can be used to quantify the performance of both the benchmark models and the solution models are the loss values and the Root Mean Squared Errors (RMSE). 

The RMSE is frequently used to measure the differences between values predicted by a model or an estimator and the values observed. The RMSE represents the square root of the mean of the square of the differences between predicted values ($\hat{Y}$) and observed values ($Y$).

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(Y_{i} - \hat{Y}_i)^2}
$$

The RMSE is always a positive value and a value of zero would indicate a perfect fit to the data.

### Project Design

I first need to explore the data in order to understand it. Correlation matrices can be a good start to understand which features will be more relevant for predicting the market price of Bitcoin. The use of Time Series will require some preprocessing of the data as I need to decide how far in the past I want to look in order to predict a point in the future. Therefore the data need to be reshaped in consequence.

Once the data preprocessing is done, I will be able to train my models. A few architecture have to be tried in order to find the best one for the solution. Also I will have to build the models for the benchmark and compare the RMSE values I get for all models I tried. Finally I will have to draw visualizations of the results.


[1]:https://bitcoin.org/bitcoin.pdf
[2]: https://ieeexplore.ieee.org/document/8126078/
[3]:https://www.jbs.cam.ac.uk/fileadmin/user_upload/research/centres/alternative-finance/downloads/2017-global-cryptocurrency-benchmarking-study.pdf
[4]:https://en.wikipedia.org/wiki/History_of_bitcoin
[5]: https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory/home