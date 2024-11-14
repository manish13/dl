import pandas as pd 
import numpy as np
# construct the technical factors using price_volume_data

# data read utils
def get_price_data():
    root = '/home/manish/USData/data/us_stocks.parquet'
    a = pd.read_parquet(root)
    return a

def get_ff3():
    root = '/home/manish/USData/FF3.csv'
    a = pd.read_csv(root, index_col=0, parse_dates=True)    
    return a

# universe construction
def get_universe(a):
    mcap = a.marketCap
    mcap=mcap[~mcap.index.duplicated(keep='last')].unstack(level=0)
    mr = mcap.rolling(63, min_periods=10).mean().rank(axis=1, ascending=False)
    u = mr<1000
    import datetime as dt
    u=u[u.index>'19920401']
    u=u[u.columns[u.sum()>0]]
    return u 

# data utils 
def get_mkt(ret, mcap):
    return (ret * np.sqrt(mcap)).sum(1).divide(np.sqrt(mcap).sum(1), axis=0)

def get_shares(mcap, prc):
    return mcap/prc 

def get_dolvol(vol, prc):
    return vol * prc

def get_data(df):
    return df[~df.index.duplicated(keep='last')].unstack(level=0)

def shapify(df, u):
    return df.reindex(columns=u.columns, index=u.index)[u]


# technical factors
def MOM_1M(ret):
    return -ret.rolling(21, min_periods=1).mean()

def ZEROTRADE(vol):
    return (vol==0).rolling(21*3, min_periods=1).sum()

def ME(mcap):
    return mcap 

def STD_DOL_VOL(vol, prc):
    return (vol*prc).rolling(21*3, min_periods=5).std()

def SEAS1A(ret):
    return ret.rolling(21, min_periods=5).mean().shift(252-10)

def BETA(ret, mkt):
    return ret.rolling(21*3, min_periods=5).cov(mkt)/mkt.rolling(21*3, min_periods=5).var()

def CHCSHO(mcap, prc):
    return (mcap/prc).diff(1)

def RVAR_MEAN(ret):
    return ret.rolling(21*3, min_periods=5).var()

def MOM6M(ret):
    return ret.rolling(21*5, min_periods=5).mean().shift(21)

def DOLVOL(vol, prc):
    return vol*prc 

def MOM60M(ret):
    return - ret.rolling(21*60, min_periods=21*12).shift(12*21)

def MOM36M(ret):
    return ret.rolling(21*36, min_periods=21*12).shift(12*21)

def TURN(vol, mcap, prc):
    return vol/mcap * prc

def STD_TURN(vol, mcap, prc):
    return (vol/mcap*prc).rolling(21*3, min_periods=5).std()

def MOM12M(ret):
    return ret.rolling(21*12, min_periods=21).shift(21)

def RVAR_CAPM(ret, mkt):
    beta = BETA(ret, mkt)
    e = ret - beta * mkt 
    return e.rolling(12*21, min_periods=21).std()

def RVAR_FF3(ret, ff3):
    beta = pd.DataFrame({BETA(ret, ff3[i]) for i in ff3.columns})
    e = ret - beta * ff3 
    return e.rolling(12*21, min_periods=21).std()

if __name__ == '__main__':
    # get price data
    a = get_price_data()
    print(a.shape, a.columns)

    # construct universe
    u = get_universe(a)
    print(u.shape)

    # get return/volume/mcap data
    r = get_data(a.changePercent, u)
    print(r.shape)
    m = get_data(a.marketCap, u)
    print(m.shape)
    v = get_data(a.volume, u)
    print(v.shape)
    c = get_data(a.close, u)
    print(c.shape)

