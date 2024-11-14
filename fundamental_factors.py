import pandas as pd
import numpy as np
import os
import json

universe_file_path = './data/universe.parquet'
underlying_file_path = './data/underlying/stocks/{}/'

base_fundamental_data_location = {
    'earningsYield': 'key-metrics-quarterly',
    'revenuePerShare': 'key-metrics-quarterly',
    'taxPayables': 'balance-sheet-statement-quarterly',
    'totalAssets': 'balance-sheet-statement-quarterly',
    'numberOfShares': 'enterprise-values-quarterly',
    'bookValuePerShare': 'key-metrics-quarterly',
    'stockPrice': 'enterprise-values-quarterly',
    'netIncomePerShare': 'key-metrics-quarterly',
    'netIncome': 'cash-flow-statement-quarterly',
    'totalStockholdersEquity': 'balance-sheet-statement-quarterly',
    'debtToEquity': 'key-metrics-quarterly',
    'depreciationAndAmortization': 'cash-flow-statement-quarterly',
    'tangibleAssetValue': 'key-metrics-quarterly',
    'operatingCashFlow': 'cash-flow-statement-quarterly',
    'totalDebt': 'balance-sheet-statement-quarterly',
    'operatingCashFlowPerShare': 'key-metrics-quarterly',
    'dividendYield': 'key-metrics-quarterly',
    'researchAndDdevelopementToRevenue': 'key-metrics-quarterly',
    'returnOnTangibleAssets': 'key-metrics-quarterly',
    'cashAtEndOfPeriod': 'cash-flow-statement-quarterly',
    'changeInWorkingCapital': 'cash-flow-statement-quarterly',
    'longTermDebt': 'balance-sheet-statement-quarterly'
}

def shapify(df, u):
    return df.reindex(index=u.index, method='ffill').reindex(columns=u.columns)

def get_universe():
    try:
        return pd.read_parquet(universe_file_path)
    except Exception as e:
        print(f"Error reading universe file: {e}")
        return None

def make_base_fundamental_data(file_name, data_name):
    u = get_universe()
    if u is None:
        return
    all_stock_df = pd.DataFrame()
    for stock in u.columns:
        file_path = os.path.join(underlying_file_path.format(stock), file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df = df[['date', data_name]]
                df['symbol'] = stock
                all_stock_df = pd.concat([all_stock_df, df], ignore_index=True)
        # print(all_stock_df.tail())
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
    all_stock_df.to_parquet(f'./data/{data_name}.parquet')


for k, v in base_fundamental_data_location.items():
    make_base_fundamental_data(v, k)
    print(f"Finished making {k}.parquet")
    df = pd.read_parquet(f'./data/{k}.parquet')
    print(df.tail())
