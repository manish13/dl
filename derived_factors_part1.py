# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:47:17 2024

@author: zouit
"""

import pandas as pd
import numpy as np
import os
import json

universe_file_path = './data/universe.parquet'
underlying_file_path = './data/underlying/stocks/{}/'

base_fundamental_data_location = {
    'netIncome': 'cash-flow-statement-quarterly', 
    'depreciationAndAmortization': 'cash-flow-statement-quarterly', 
    'operatingCashFlow': 'cash-flow-statement-quarterly', 
    'earningsYield': 'key-metrics-quarterly',
    'revenuePerShare': 'key-metrics-quarterly',
    'taxPayables': 'balance-sheet-statement-quarterly',
    'totalAssets': 'balance-sheet-statement-quarterly',
    'numberOfShares': 'enterprise-values-quarterly',
    'bookValuePerShare': 'key-metrics-quarterly',
    'stockPrice': 'enterprise-values-quarterly',
    'netIncomePerShare': 'key-metrics-quarterly',
    'totalStockholdersEquity': 'balance-sheet-statement-quarterly',
    'debtToEquity': 'key-metrics-quarterly',
    'tangibleAssetValue': 'key-metrics-quarterly',
    'totalDebt': 'balance-sheet-statement-quarterly',
    'operatingCashFlowPerShare': 'key-metrics-quarterly',
    'dividendYield': 'key-metrics-quarterly',
    'researchAndDdevelopementToRevenue': 'key-metrics-quarterly',
    'returnOnTangibleAssets': 'key-metrics-quarterly',
    'cashAtEndOfPeriod': 'cash-flow-statement-quarterly',
    'changeInWorkingCapital': 'cash-flow-statement-quarterly',
    'longTermDebt': 'balance-sheet-statement-quarterly',
    'priceToSalesRatio' :'key-metrics-quarterly'
}

derived_growth_factors = [
    "EP", "CFP", "DY", "RDM", "RNA", "CASH", "ACC", "NOA", "ROA", "SP", "TES",
    "BM", "PM", "ROE", "Leverage", "ATO", "DEPR", "CASHDEBT"
]

def shapify(df, u):
    daily = df.reindex(index=u.index, method='ffill').reindex(columns=u.columns)
    monthly = daily.resample('M').last()
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
    all_stock_df = shapify(all_stock_df, u)
    try:
        all_stock_df = all_stock_df.astype(np.float64)
    except OverflowError:
        all_stock_df = all_stock_df.apply(pd.to_numeric, errors='coerce')
        # all_stock_df = all_stock_df.div(1e6)
        all_stock_df = all_stock_df.astype(np.float64)
    os.makedirs('./data/raw_factors', exist_ok=True)    
    all_stock_df.to_parquet(f'./data/raw_factors/{data_name}.parquet')
    
if __name__ == '__main__':

    for k, v in base_fundamental_data_location.items():
        try:
            make_base_fundamental_data(v, k)
            print(f"Finished making {k}.parquet")
            #df = pd.read_parquet(f'./data/raw_factors/{k}.parquet')
            #print(df.tail())
        except Exception as e:
            print(f"Error making {k}.parquet: {e}")
            continue
            
    u = get_universe()
    u = u.drop(['CYOU','FENG','MAGS'], axis = 1)
    os.makedirs('./data/derived_factors', exist_ok=True)    
    for factor in derived_growth_factors :
        try :
            if factor == "EP":
                EP = pd.read_parquet('./data/raw_factors/earningsYield.parquet')
                EP.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "CFP":
                CFP = pd.read_parquet('./data/raw_factors/operatingCashFlowPerShare.parquet')
                CFP.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "DY":
                DY = pd.read_parquet('./data/raw_factors/dividendYield.parquet')
                DY.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "RDM":
                RDM = pd.read_parquet('./data/raw_factors/researchAndDdevelopementToRevenue.parquet')
                RDM.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "RNA":
                RNA = pd.read_parquet('./data/raw_factors/returnOnTangibleAssets.parquet')
                RNA.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "CASH":
                CASH = pd.read_parquet('./data/raw_factors/cashAtEndOfPeriod.parquet')
                CASH.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "ACC":
                ACC = pd.read_parquet('./data/raw_factors/changeInWorkingCapital.parquet')
                ACC.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "NOA":
                NOA = pd.read_parquet('./data/raw_factors/operatingCashFlow.parquet')
                NOA.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "ROA":
                ROA = pd.read_parquet('./data/raw_factors/returnOnTangibleAssets.parquet')
                ROA.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "SP":
                SP = pd.read_parquet('./data/raw_factors/priceToSalesRatio.parquet')
                SP.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            
            elif factor == "TES":
                TES = pd.read_parquet('./data/raw_factors/taxPayables.parquet')
                TES.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
                
            elif factor == "BM":
                num = pd.read_parquet('./data/raw_factors/bookValuePerShare.parquet')
                den = pd.read_parquet('./data/raw_factors/stockPrice.parquet')
                BM = num / den
                BM.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")
            elif factor == "PM":
                num = pd.read_parquet('./data/raw_factors/netIncomePerShare.parquet')
                den = pd.read_parquet('./data/raw_factors/revenuePerShare.parquet').values
                PM = num / den
                PM.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")

            elif factor == "ROE":
                num = pd.read_parquet('./data/raw_factors/netIncome.parquet')
                den = pd.read_parquet('./data/raw_factors/totalStockholdersEquity.parquet')
                ROE = num / den
                ROE.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")

            elif factor == "Leverage":
                leverage = 1 / pd.read_parquet('./data/raw_factors/debtToEquity.parquet')
                Leverage = leverage
                Leverage.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")

            elif factor == "ATO":
                numerator = pd.read_parquet('./data/raw_factors/revenuePerShare.parquet') * pd.read_parquet('./data/raw_factors/numberOfShares.parquet')
                denominator = pd.read_parquet('./data/raw_factors/totalAssets.parquet')
                ATO = numerator / denominator
                ATO.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")

            elif factor == "DEPR":
                num = pd.read_parquet('./data/raw_factors/depreciationAndAmortization.parquet')
                den = pd.read_parquet('./data/raw_factors/tangibleAssetValue.parquet')
                DEPR = num / den
                DEPR.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")

            elif factor == "CASHDEBT":
                num = pd.read_parquet('./data/raw_factors/operatingCashFlow.parquet')
                den = pd.read_parquet('./data/raw_factors/totalDebt.parquet')
                CASHDEBT = num / den
                CASHDEBT.to_parquet(f'./data/derived_factors/{factor}.parquet')
                print(f"Finished making {factor}.parquet")

                
        except Exception as e:
            print(f"Error making {factor}.parquet: {e}")
            

            
            
        







