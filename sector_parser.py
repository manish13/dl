root = '/home/manish/USData/data/stocks/'
import pandas as pd 
import os 

def get_group_data(sym: str) -> pd.Series:
    a = pd.read_json(root+sym+'/company-profile')
    return pd.Series({'industry': a.T.industry.iloc[1], 'sector': a.T.sector.iloc[1]})

l = sorted(os.listdir(root))
success = 0
failure = 0
data = {}
for i, sym in enumerate(l):
    print(f'{sym}: {i}/{len(l)}')
    try:
        data[sym] = get_group_data(sym)
        success += 1
    except Exception as e:
        print(f'{sym} failed')
        failure += 1
        # print(e)
        # raise ValueError(f'{sym} failed!')
    print(success, failure, len(l))

d = pd.concat(data, axis=1).T

d.to_parquet('sector_industry.parquet')