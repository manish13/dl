import pandas as pd
import numpy as np
import matplotlib as mpl
import os

root: str = '/home/manish/USData/data/stocks/'
sym: str = 'A'

sf = ['ratios-quarterly', 'key_metrics-quarterly', 
      'income-statement-quarterly', 'historical-rating-quarterly',
      'historical-discounted-cash-flow-quarterly',
      'historical-daily-discounted-cash-flow',
      'financial-growth-quarterly',
      'enterprise-value-quarterly',
      'cash-flow-statement-quarterly',
      'balance-sheet-statement-quarterly'] 
      
core = ['historical-price-full','stock_dividend', 'stock_split',
        'historical-market-capitalization', ]


gics = ['company-profile']


def get_price_data(sym: str):
    # price data
    fname = f'/home/manish/USData/data/stocks_cleaned/{sym}.parquet'
    if not os.path.isfile(fname):
        a=pd.read_json(root+sym+'/historical-price-full')
        df = []
        for r in a.historical.index:
            df.append(pd.Series(a.historical[r])[['date','close', 'unadjustedVolume', 'changePercent']])
        df = pd.concat(df, axis=1).T.set_index('date')
        df.index = pd.to_datetime(df.index)

        f = open(root+sym+'/historical-market-capitalization', 'r')
        mcap = pd.read_json(f.readlines()[0]).set_index('date')[['marketCap']]
        mcap.index = pd.to_datetime(mcap.index)

        data = df.join(mcap).rename(columns={'unadjustedVolume': 'volume'}).sort_index()
        data.to_parquet(fname)

# l = sorted(os.listdir(root))
# success = 0
# failure = 0
# for i, sym in enumerate(l):
#     print(f'{sym}: {i}/{len(l)}')
#     try:
#         get_price_data(sym)
#         success += 1
#     except Exception as e:
#         print(f'{sym} failed')
#         failure += 1
#         # print(e)
#         # raise ValueError(f'{sym} failed!')
#     print(success, failure, len(l))

l = os.listdir(f'/home/manish/USData/data/stocks_cleaned/')
df = {}
for i, sym in enumerate(l):
    sym = '.'.join(sym.split('.')[:-1])
    print(sym, i, len(l))
    df[sym] = pd.read_parquet(f'/home/manish/USData/data/stocks_cleaned/{sym}.parquet')
a=pd.concat(df)
a.reset_index(level=0, names=['asset'])