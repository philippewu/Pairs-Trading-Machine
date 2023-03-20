# Pairs-Trading-Machine
## Introduction

This Pairs Trading Machine allows the customization and testing of the classical pairs trading model on historical stock data.


## Default Model
Current implementation closely follows Gatev, Goetzmann, and Rouwenhorst 1998 (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=141615) for the model and default parameters. The pair formation and trading period has been altered to 1 year. Data frequency was also limited to daily prices.


## Compatible Datasets
The model takes in the closing day stock prices for every trading day in a specified time period. The table must have seven columns named: year, date, TICKER, PRC, rel_PRC, eff_PRC, and eff_rel_PRC. These consist of int "YYYY", int "YYYYMMDD", string for the stock ticker, float for price, float for relative price, float for effective price, and float for effective relative price. Stock price data can be found in online sources and relative price, effective price, and effective relative price can be derived from the data.

To calculate the relative price, first choose the training and trading period length T. Over that specified period of time T, the relative price is defined as p_t/p_0 for all t in T where t is an interval of time. The effective price and effective relative prices normalize price values over stock splits. A sample formatted dataset of daily stock prices in 2002 and 2003 from the S&P 500 where T = 1 year is included (sample_data.csv). Data sets used with the program should be formatted identically.


## Usage
The model may be run with three functions: period_profits, det_period_profits, and cp_period_profits. The latter two gives detailed information of trades.

To use the functions period_profits and det_period_profits, the following parameters must be specified:
- X: the dataset
- year: the year used for pair formulation
- trigger_size: the trigger_size * avg_pair_distance is the minimum distance to open a position (2 is default)
- stop_size: the stop_size * avg_pair_distance is the maximum loss before closing a position (0 is default and represents no stop)
- mode: 0 sets the machine to trade in the next period after the pair formulation period. 1 sets the machine to trade in a specified period (trade_year)
- trade_year: put 'na' if mode=0
- pair_num: the number of pairs to use for trading (default is 10)

cp_period_profits allows the use of customized sets of stocks as pairs. To use cp_period_profits, an additional parameter must be specified:
- cp: a Nx2 array where each row consists of the two tickers that make up a pair

## Interpretation of Results
The results are calculated assuming that investments are divided equally among each pair and a constant amount is invested in the opening of each trade. In other words, realized gains and losses are not reinvested.

For each pair in the defined set, four entries are reported
- Pairs: the tickers that make up the pair
- Trades: a record of trades. Each item in the list makes up a unique completed trade. Each completed trade has three numbers. The first number indicates the opening time t, the second number indicates the closing time t, and the third number indicates the direction of the trade (0 for shorting the first ticker and 1 for shorting the second ticker). If the closing number is >=1000, the position was closed due to time expiration.
- Trade profits: a list of returns for each trade whose position corresponds to that of the record of trades
- Pair profits: the sum of the trade profits for a specific pair

Finally, the function also reports the overall return from the portfolio.

## Customization
The model can be customized by changing the parameters of the functions. These include
- dataset (universe of stocks considered)
- pair formation year
- trigger size (or the mimimum distance requirement to open a position, set at 2 in the paper)
- stop size (maximum loss before closing a position, set at 0 or none in the paper)
- custom selected pairs

Other aspects of the model will require recoding functions. These include
- Changing the distance metric
- Changing the training and trading period lengths
