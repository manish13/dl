import pandas as pd 
import numpy as np
# construct the technical factors using price_volume_data

root = '/home/manish/USData/data/us_stocks.parquet'

a = pd.read_parquet(root)

# construct the universe
mcap = a.marketCap
mcap=mcap[~mcap.index.duplicated(keep='last')].unstack(level=0)
mr = mcap.rolling(63, min_periods=10).mean().rank(axis=1, ascending=False)
u = mr<1000
import datetime as dt
u=u[u.index>'19920401']
u=u[u.columns[u.sum()>0]]
print(u.shape)


# get return/volume/mcap data

