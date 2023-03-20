import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.stats import gmean
import statsmodels.api as sm



## FUNCTIONS TO PROCESS DATA: the machine uses relative which must be derived from the data
# creates a relative price column in the data
# relative price is defined as [price at time t]/[price at time 0] for all t in T
def rel_prc(X, col_name, new_name):
	X[new_name] = X[col_name] / X[col_name].values[0]
	return X



## MATHEMATICAL FUNCTIONS
# calculates the standard deviation
def std(X):
	X["std"] = np.sqrt(np.sum((X.PRC.values - np.mean(X.PRC.values)) ** 2) / len(X.PRC.values))
	return X


# calculates geometric mean rate of return
def gmean_rate(X):
	return gmean(X + 1) - 1



## TESTING FUNCTIONS: useful for testing various distance metrics
# given a ticker and a year, return the n closest stocks in distance
# distance is defined by the sum of squared deviations in price distances
def prc_dist_raw(X, TCK, year, num_pairs):
	yr_df = X.loc[X["year"]==year,["year","date","TICKER","eff_PRC","eff_rel_PRC"]]
	yr_tck = np.array(list(set(yr_df["TICKER"].values)))
	rel_PRC = yr_df[["TICKER","eff_rel_PRC"]].values
	dist_yr = []
	test_stk = yr_df.loc[yr_df["TICKER"]==TCK,:]["eff_rel_PRC"].values
	trading_days = rel_PRC[rel_PRC[:,0]=="WMT",:]

	for i in yr_tck:
		comp_prc = rel_PRC[rel_PRC[:,0]==i,:]
		if len(comp_prc) == len(trading_days):
			dist = np.sum(np.sqrt((test_stk - comp_prc[:,1].astype(float)) ** 2)) / len(trading_days)
		else:
			dist = np.inf
		dist_yr.append([i,dist])

		dist_yr = np.array(dist_yr)
		top_pairs = dist_yr[dist_yr[:,1].argsort()]

		return top_pairs[0:num_pairs,:]



## PAIR FORMATION FUNCTIONS: these functions allow the model to choose stocks to trade
# given the data and a specified year, return an array of all stocks and their relative price column
def prc_dist_data(X, year):
	rel_PRC = X.loc[X["year"]==year,["TICKER","eff_rel_PRC"]].values
	yr_tck = np.array(list(set(rel_PRC[:,0])))
	trading_days = rel_PRC[rel_PRC[:,0]=="WMT",:]

	dataset = []
	data_tck = []
	for i in yr_tck:
		comp_prc = rel_PRC[rel_PRC[:,0]==i,:]
		if (len(comp_prc) == len(trading_days)) and ~np.isnan(comp_prc[:,1].astype(float)).any():
			dataset.append(comp_prc)
			data_tck.append(i)

	return np.array(dataset), np.array(data_tck)


# given a pair, find the pair distance over a given year
def prc_dist(X, pair1, pair2, year):
    rel_PRC = X.loc[X["year"]==year,["TICKER","eff_rel_PRC"]].values
    trading_days = rel_PRC[rel_PRC[:,0]=="WMT",:]
    pair1_rel_prc = rel_PRC[rel_PRC[:,0]==pair1,:]
    pair2_rel_prc = rel_PRC[rel_PRC[:,0]==pair2,:]
    dist = np.sum(np.sqrt((pair1_rel_prc[:,1].astype(float) - pair2_rel_prc[:,1].astype(float)) ** 2)) / len(trading_days)
    return dist


# given all relative prices, find the nearest neighbor graph of all stock pairs
def prc_dist_nn(data):
    M = kneighbors_graph(data, len(data) - 1, mode='distance').toarray()
    return M


# return n best pairs within a given year
def top_pairs(M, num_pairs, tickers):
    M = np.triu(M)
    M = np.where(M == 0, np.inf, M)
    top_pairs = np.argwhere(np.isin(M, np.sort(M.flatten())[0:num_pairs]))
    
    shp = np.shape(top_pairs)
    mapped = np.empty(shp, dtype='<U5')
    for index, x in np.ndenumerate(top_pairs):
        mapped[index[0], index[1]] = tickers[x]
        
    dist_map = []
    for i in top_pairs:
        dist_map.append(M[i[0]][i[1]])

    dist_map = np.array(dist_map).reshape(num_pairs,1)
        
    return top_pairs, mapped, dist_map



## MACHINE FUNCTIONS: these functions operate the trades
# given the year and the pair of stocks, return the trading profits of that period
def backtesting(X, year, pair1, pair2, trigger_size, stop_size, mode, trade_year):
    # model dependent parameter initialization
    if mode == 0:
        trading_days = len(X.loc[(X["TICKER"]=="WMT")*(X["year"]==year)])
        yr_data = X.loc[X["year"]==year,:]
        p1 = yr_data.loc[X["TICKER"]==pair1,['eff_PRC','eff_rel_PRC']].values
        p2 = yr_data.loc[X["TICKER"]==pair2,['eff_PRC','eff_rel_PRC']].values
        last_day = np.round(len(p1) / 1).astype(int)
    else:
        trading_days = len(X.loc[(X["TICKER"]=="WMT")*(X["year"]==trade_year)])
        yr_data = X.loc[X["year"]==trade_year,:]
        p1 = yr_data.loc[X["TICKER"]==pair1,['eff_PRC','eff_rel_PRC']].values
        p2 = yr_data.loc[X["TICKER"]==pair2,['eff_PRC','eff_rel_PRC']].values
        last_day = np.round(len(p1) / 1).astype(int)
    
    if (len(p1) != trading_days) or (len(p2) != trading_days):
        return np.array([]), np.array([]), 0
    
    div = (p1[:,1] - p2[:,1]).flatten()[:last_day]
    distance = prc_dist(X, pair1, pair2, year - 1)
    
    if stop_size == 0:
        stop = np.zeros(len(div)) > 1
        trigger = np.where(np.absolute(div) > distance * trigger_size)
    else:
        stop = np.absolute(div) > distance * stop_size
        trigger = np.where((np.absolute(div) > distance * trigger_size) * (np.absolute(div) < distance * stop_size))
    
    # trading machine
    cross = div > 0
    close_day = 0
    trades = []
    
    while True:
        open_days = trigger[0][trigger[0] > close_day]
        if len(open_days) > 0:
            open_day = open_days[0]
            
            if cross[open_day]:
                a = np.where(~cross[open_day:])[0]
                b = np.where(stop[open_day:])[0]
                if (len(a) > 0) and (len(b) > 0):
                    close_day = min(a[0], b[0])
                    if close_day == b[0]:
                        close_day += open_day
                        trades.append([open_day, close_day, cross[open_day]])
                        break
                elif len(a) > 0:
                    close_day = a[0]
                elif len(b) > 0:
                    close_day = b[0]
                    close_day += open_day
                    trades.append([open_day, close_day, cross[open_day]])
                    break
                else:
                    close_day = 1000
            else:
                a = np.where(cross[open_day:])[0]
                b = np.where(stop[open_day:])[0]
                if (len(a) > 0) and (len(b) > 0):
                    close_day = min(a[0], b[0])
                    if close_day == b[0]:
                        close_day += open_day
                        trades.append([open_day, close_day, cross[open_day]])
                        break
                elif len(a) > 0:
                    close_day = a[0]
                elif len(b) > 0:
                    close_day = b[0]
                    close_day += open_day
                    trades.append([open_day, close_day, cross[open_day]])
                    break
                else:
                    close_day = 1000
            
            close_day += open_day
            trades.append([open_day, close_day, cross[open_day]])
        else:
            break
        
    trades = np.array(trades)
    
    # return calculation
    pi = []
    for i in trades:
        if i[2] == 1:
            if i[1] < 1000:
                pi_pt = (p1[:,0][i[0]] / p1[:,0][i[1]]) + (p2[:,0][i[1]] / p2[:,0][i[0]]) - 2
            else:
                pi_pt = (p1[:,0][i[0]] / p1[:,0][last_day - 1]) + (p2[:,0][last_day - 1] / p2[:,0][i[0]]) - 2
        else:
            if i[1] < 1000:
                pi_pt = (p1[:,0][i[1]] / p1[:,0][i[0]]) + (p2[:,0][i[0]] / p2[:,0][i[1]]) - 2
            else:
                pi_pt = (p1[:,0][last_day - 1] / p1[:,0][i[0]]) + (p2[:,0][i[0]] / p2[:,0][last_day - 1]) - 2
        pi.append(pi_pt)
    
    pi = np.array(pi)
    tot_pi = np.sum(pi)
    
    # trades is a 2d array with information on the duration and execution of each trade for each stock pair
    # pi is an array of the profits for each pair
    # tot_pi is the total profits of the entire portfolio
    return trades, pi, tot_pi



## MODEL FUNCTIONS: allow the customization of the model
# model parameters:
# X - the dataset used
# year - the pair formulation time specification
# trigger_size - when a trading position is opened (default is 2)
# stop_size - the conditions where the model should cut losses on a trade (default is 0 or no loss cutting)
# mode - (either 0 or 1) determines whether the machine trades in the next year (mode 0) or in a specified year
# (mode 1) given by the trade_year
# pair_num - number of pairs taken to trade from pair formulation
# cp - list of custom pairs that can be used instead of the normal pair formulation procedure

# period profits returns the profits of the given period with the specified model
def period_profits(X, year, trigger_size, stop_size, mode, trade_year):
    prc, tck = prc_dist_data(X, year)
    M = prc_dist_nn(prc[:,:,1].astype(float))
    pair_ind, pair_tck, pair_dist = top_pairs(M, 10, tck)
    per_pi = 0
    dead_pairs = 0
    
    for i in np.arange(0,len(pair_tck)):
        days, ind_pi, pair_pi = backtesting(X, year + 1, pair_tck[i][0], pair_tck[i][1], trigger_size, stop_size, mode, trade_year)
        per_pi += pair_pi
        if pair_pi == 0:
            dead_pairs += 1
    
    if (len(pair_tck) - dead_pairs) == 0:
        per_pi = 0
    else:
        per_pi = per_pi / (len(pair_tck) - dead_pairs)
    
    return per_pi


# period profits with details
def det_period_profits(X, year, trigger_size, stop_size, mode, trade_year, pair_num):
    prc, tck = prc_dist_data(X, year)
    M = prc_dist_nn(prc[:,:,1].astype(float))
    pair_ind, pair_tck, pair_dist = top_pairs(M, pair_num, tck)
    per_pi = 0
    dead_pairs = 0
    
    info = []
    
    for i in np.arange(0,len(pair_tck)):
        days, ind_pi, pair_pi = backtesting(X, year + 1, pair_tck[i][0], pair_tck[i][1], trigger_size, stop_size, mode, trade_year)
        per_pi += pair_pi
        info.append([pair_tck[i], days, ind_pi.round(4), pair_pi])
        if pair_pi == 0:
            dead_pairs += 1
    
    if (len(pair_tck) - dead_pairs) == 0:
        per_pi = 0
    else:
        per_pi = per_pi / (len(pair_tck) - dead_pairs)
    
    return info, per_pi


# period profits with custom pair choices (if mode is 1, year determines the distance metric)
def cp_period_profits(X, cp, year, trigger_size, stop_size, mode, trade_year, pair_num):
    pair_tck = cp
    per_pi = 0
    dead_pairs = 0
    
    info = []
    
    for i in np.arange(0,len(pair_tck)):
        days, ind_pi, pair_pi = backtesting(X, year + 1, pair_tck[i][0], pair_tck[i][1], trigger_size, stop_size, mode, trade_year)
        per_pi += pair_pi
        info.append([pair_tck[i], days, ind_pi.round(4), pair_pi])
        if pair_pi == 0:
            dead_pairs += 1
    
    if (len(pair_tck) - dead_pairs) == 0:
        per_pi = 0
    else:
        per_pi = per_pi / (len(pair_tck) - dead_pairs)
    
    return info, per_pi