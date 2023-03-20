from machine import det_period_profits
import pandas as pd
from IPython.display import display

# model setup with defaults using standard pair formation in 2002 and trading in 2003
data = pd.read_csv('sample_data.csv')
details, pi = det_period_profits(data, 2002, 2, 0, 0, 'na', 10)
df = pd.DataFrame(details).rename({0:"Pairs",1:"Trades",2:"Trade Profits",3:"Pair Profit"}, axis=1)

# trading results
print("2002-2003:", pi)
display(df)