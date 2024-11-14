import pandas as pd, numpy as np
import technical_factors as techfac 


Technical = {
'MOM_1M': 'short term reversl 1-1 month',
'ZEROTRADE': 'number of zero-trading days (3 months)',
'ME': 'Market Equity',
'STD_DOL_VOL': 'Std of dollar trading volume (3 motnhs)',
'SEAS1A': 'seaonality',
'BETA': 'Beta (3m)',
'CHCSHO': 'change in shares outstanding',
'RVAR_MEAN': 'return variance (3 month)',
'MOM6M': 'momentum 6 m (2-6 months)',
'DOLVOL': 'dollar trading volume',
'MOM60M': 'long term reversal (13-60 months)',
'MOM36M': 'momemtum 36 m (13-36 month)',
'TURN': 'shares turnover',
'STD_TURN': 'std of share turnover (3 month)',
'MOM12M': 'momentum (2-13 months)',
'RVAR_CAPM': 'CAPM residual variance (3 months)',
'RVAR_FF3': 'FF3 residual variance (3 month)',

'MAXRET': 'maximum daily return',
'ABR': 'abnormal returns around earnings dates',
'BASPREAD': 'bid-ask spread (3m)',
'ALM': 'asset liquidity',
'NI': 'net stock issues',
'ILL': 'illiquidity (3 months)',
'ME_IA': 'industry-adjusted market equity',
'ATO': 'asset turnover',
}

Fundamental ={
'SUE': 'unexpected quarterly earnings',
'BM_IA': 'industry adjusted BM',
'BM': 'book to market',
'NOA': 'net operating assets',
'ROA': '',
'CFP': '',
'PM': '',
'LGR': '',
'CASH': '',
'LEV': '',
'RD_SALE': '',
'SP': '',
'EP': '',
'RNA': '',
'AGR': '',
'SGR': '',
'OP': '',
'ROE': '',
'HERF': '',
'ADM': '',
'HIRE': '',
'PSCORE': '',
'DEPR': '',
'DY': '',
'RSUP': '',
'CASHDEBT': '',
'CHPM': '',
'GMA': '',
'PCTACC': '',
'CINVEST': '',
'NINCR': '',
'ACC': '',
'GRLTNOA': '',
'CHTX': ''
}

def make_universe():
    u = techfac.get_universe(techfac.get_price_data())
    u.to_parquet('data/universe.parquet')

def make_returns():
    a = techfac.get_price_data()
    r = techfac.get_data(a.close).pct_change()
    r[r.abs()>0.2] = np.nan  # this is adhoc cleaning but should suffice for training
    u = CharacterisitcMaker().universe()
    r.reindex(columns=u.columns).to_parquet('data/returns.parquet')

def make_price():
    a = techfac.get_price_data()
    p = techfac.get_data(a.close).ffill(limit=1)
    u = CharacterisitcMaker().universe()
    p.reindex(columns=u.columns).to_parquet('data/price.parquet')

def make_volume():
    a = techfac.get_price_data()
    p = techfac.get_data(a.volume).fillna(0)
    u = CharacterisitcMaker().universe()
    p.reindex(columns=u.columns).to_parquet('data/volume.parquet')

def make_mktcap():
    a = techfac.get_price_data()
    p = techfac.get_data(a.marketCap).ffill(limit=1)
    u = CharacterisitcMaker().universe()
    p.reindex(columns=u.columns).to_parquet('data/mktcap.parquet')

def make_ff3():
    prc = pd.read_parquet('data/price.parquet')
    a = pd.read_csv('data/FF3.csv', index_col=0, parse_dates=True)
    a.reindex(index=prc.index).to_parquet('data/ff3.parquet')


input_map = {
    'MOM_1M': ['returns'],
    'ZEROTRADE': ['volume'],
    'ME': ['mktcap'],
    'STD_DOL_VOL': ['volume', 'price'],
    'SEAS1A': ['returns'],
    'BETA': ['returns', 'market'],
    'CHCSHO': ['mktcap', 'price'],
    'RVAR_MEAN': ['returns'],
    'MOM6M': ['returns'],
    'DOLVOL': ['volume', 'price'],
    'MOM60M': ['returns'],
    'MOM36M': ['returns'],
    'TURN': ['volume', 'mktcap', 'price'],
    'STD_TURN': ['volume', 'mktcap', 'price'],
    'MOM12M': ['returns'],
    'RVAR_CAPM': ['returns', 'market'],
    'RVAR_FF3': ['returns', 'ff3'],
}
class CharacterisitcMaker:
    def __init__(self, name: str = None):
        assert name in input_map
        self.name = name # name of the characteristic

    def universe(self):
        return pd.read_parquet('data/universe.parquet')

    def returns(self):
        return pd.read_parquet('data/returns.parquet')
    
    def price(self):
        return pd.read_parquet('data/price.parquet')
    
    def volume(self):
        return pd.read_parquet('data/volume.parquet')
    
    def mktcap(self):
        return pd.read_parquet('data/mktcap.parquet')
    
    def ff3(self):
        return pd.read_parquet('data/ff3.parquet')[['Mkt-RF','SMB','HML']]
    
    def market(self):
        ff3 = pd.read_parquet('data/ff3.parquet')
        return ff3['Mkt-RF'] + ff3['RF']
    
    def get_params(self):
        # for self.name return all parameters required
        return [getattr(self, i)() for i in input_map[self.name]]

    def __call__(self, save=False):
        if self.name in Technical:
            feature = techfac.shapify(getattr(techfac, self.name)(*self.get_params()), self.universe())
            if save:
                feature.to_parquet(f'/data/{self.name}.parquet')
            return feature
        else:
            raise NotImplementedError('Fundamental factors not implemented yet!')
        

if __name__ == '__main__':
    # make_universe()
    make_ff3()
    # df = CharacterisitcMaker(name='MOM_1M')()
    
    

