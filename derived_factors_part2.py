import pandas as pd
import numpy as np
import os
import json

universe_file_path = './data/universe.parquet'
underlying_file_path = './data/underlying/stocks/{}/'



"""
# Direct file loading
SUE = f_surprises(pd.read_parquet(p + "earningsYield.parquet"))
RSUP = f_surprises(pd.read_parquet(p + "revenuePerShare.parquet"))
                PM = np.array(stock_table['netIncomePerShare']) / np.array(stock_table['revenuePerShare'])
"""
growth_fundamental_data_location = {
    'totalAssets': 'balance-sheet-statement-quarterly',
    'longTermDebt': 'balance-sheet-statement-quarterly',
    'earningsYield': 'key-metrics-quarterly',
    'numberOfShares': 'enterprise-values-quarterly',
    'stockPrice': 'enterprise-values-quarterly',
    'netIncomePerShare': 'key-metrics-quarterly',
    'priceToSalesRatio' : 'key-metrics-quarterly',
    'revenuePerShare': 'key-metrics-quarterly'}
    
derived_growth_factors = ['GRLTNOA','LGR', 'NINCR', 'CHCSHO', 'CHPM', 'SGR']

def growth (df):
    df = df.apply(pd.to_numeric, errors='coerce')
    output = pd.DataFrame(np.zeros(df.shape), columns=df.columns, index=df.index)
    n = df.shape[0]
    for stock in df.columns:
        aux = (df[stock][1:n].reset_index(drop=True) - df[stock][0:(n-1)].reset_index(drop=True)) / df[stock][0:(n-1)].reset_index(drop=True)
        output[stock][0] = 0
        output[stock][1:n] = aux.values
    return output 

def f_surprises (df, qt = 4 ,qt2 = 8): 
    df = df.apply(pd.to_numeric, errors='coerce')
    output = pd.DataFrame(np.zeros(df.shape), columns=df.columns, index=df.index)
    n = df.shape[0]
    print(n)
    for stock in df.columns:
        dif =  - df[stock][0:(n-qt+1)].reset_index(drop=True) + df[stock][(qt-1) : n].reset_index(drop=True)
        length_differences = n - qt + 1 
        length_variances = n - qt - qt2 + 2
        print(length_variances)
        variances= np.zeros(length_variances)
        for i in range(length_variances) :
            print(i)
            variances[i] = np.std(np.array(dif[i:(qt2+i)]))
        output[stock][0:(qt2)] = np.zeros(qt2)
        output[stock][qt2:n] =  dif[(qt2-1):(len(dif))] / variances
        
    return output
    
  
def shapify(df, u, freq = 'M'):
    daily = df.reindex(index=u.index, method='ffill').reindex(columns=u.columns)
    monthly = daily.resample(freq).last()
    return monthly

def get_universe():
    try:
        return pd.read_parquet(universe_file_path)
    except Exception as e:
        print(f"Error reading universe file: {e}")
        return None

def make_base_fundamental_data(file_name, data_name):
    u = get_universe()
    u = u.drop(['CYOU','FENG','MAGS'], axis = 1)
    if u is None:
        return
    all_stock_df = pd.DataFrame()
    for stock in u.columns:
        file_path = os.path.join(underlying_file_path.format(stock), file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                try :
                    df = df[['date', data_name]]
                except:
                    print(f" {stock} missing {data_name}")
                df['symbol'] = stock
                all_stock_df = pd.concat([all_stock_df, df], ignore_index=True)
                #print(df)
        #print(all_stock_df.tail())
    #print(all_stock_df.tail())
    all_stock_df = all_stock_df.pivot(index='date', columns='symbol', values=data_name)
    all_stock_df.index = pd.to_datetime(all_stock_df.index)
    all_stock_df = all_stock_df.ffill()
    all_stock_df = shapify(all_stock_df, u,'Q')
    try:
        all_stock_df = all_stock_df.astype(np.float64)
    except OverflowError:
        all_stock_df = all_stock_df.apply(pd.to_numeric, errors='coerce')
        # all_stock_df = all_stock_df.div(1e6)
        all_stock_df = all_stock_df.astype(np.float64)
    all_stock_df.to_parquet(f'./data/raw_factors/{data_name}_Q.parquet')

if __name__ == '__main__':
    # generate raw data quarterly for growth factors :
    for k, v in growth_fundamental_data_location.items():
        try:
            make_base_fundamental_data(v, k)
            print(f"Finished making {k}_Q.parquet")
            #df = pd.read_parquet(f'./data/{k}_Q.parquet')
            #print(df.tail())
        except Exception as e:
            print(f"Error making {k}.parquet: {e}")
            continue
    #  Derived factors quarterly and resample it monthly :    
    u = get_universe()
    u = u.drop(['CYOU','FENG','MAGS'], axis = 1)
    for factor in  derived_growth_factors:
      try:
        if factor ==  'GRLTNOA':
           GRLTNOA =  pd.read_parquet('./data/raw_factors/totalAssets_Q.parquet')
           GRLTNOA = growth(GRLTNOA)
           GRLTNOA = shapify(GRLTNOA,u,'M')
           GRLTNOA.to_parquet(f'./data/derived_factors/{factor}.parquet')
           print(f"Finished making {factor}.parquet")
        elif factor ==  'LGR':
           LGR =  pd.read_parquet('./data/raw_factors/longTermDebt_Q.parquet')
           LGR = growth(LGR)
           LGR = shapify(LGR,u,'M')
           LGR.to_parquet(f'./data/derived_factors/{factor}.parquet')
           print(f"Finished making {factor}.parquet")
        elif factor ==  'NINCR':
           NINCR =  pd.read_parquet('./data/raw_factors/earningsYield_Q.parquet')
           NINCR = growth(NINCR)
           NINCR = shapify(NINCR,u,'M')
           NINCR.to_parquet(f'./data/derived_factors/{factor}.parquet')     
           print(f"Finished making {factor}.parquet")
        elif factor ==  'CHCSHO':
           CHCSHO =  pd.read_parquet('./data/raw_factors/numberOfShares_Q.parquet')
           CHCSHO = growth(CHCSHO)
           CHCSHO = shapify(CHCSHO,u,'M')
           CHCSHO.to_parquet(f'./data/derived_factors/{factor}.parquet')
           print(f"Finished making {factor}.parquet")
             
        elif factor ==  'CHPM':
           num =  pd.read_parquet('./data/raw_factors/netIncomePerShare_Q.parquet')
           den =  pd.read_parquet('./data/raw_factors/revenuePerShare_Q.parquet')
           PM = num / den
           CHPM = growth(PM)
           CHPM = shapify(CHPM,u,'M')
           CHPM.to_parquet(f'./data/derived_factors/{factor}.parquet') 
           print(f"Finished making {factor}.parquet")
        elif factor ==  'SGR':
           num =  pd.read_parquet('./data/raw_factors/stockPrice_Q.parquet')
           den =  pd.read_parquet('./data/raw_factors/priceToSalesRatio_Q.parquet')
           SGR = num /den
           SGR = growth(SGR)
           SGR = shapify(SGR,u,'M')
           SGR.to_parquet(f'./data/derived_factors/{factor}.parquet')
           print(f"Finished making {factor}.parquet")  
      except Exception as e:
            print(f"Error making {factor}.parquet: {e}")
           
           
           
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
           
    
        
        
