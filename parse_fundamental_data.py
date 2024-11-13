# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:21:14 2024

@author: zouit
"""

import os
import pandas as pd
import json
import numpy as np

#%% Auxiliary functions for fundamental data construction
def f_surprises (eps, qt = 4 ,qt2 = 8):
    n = len(eps)
    dif =  np.array(eps[0:(n-qt+1)]) - np.array(eps[(qt-1) : n]) 
    
    lentgth_differences = n - qt + 1 
    length_variances = n - qt - qt2 + 2
    variances= np.zeros(length_variances)
    
    for i in range(length_variances) :
        variances[i] = np.std(dif[i:(qt2+i)])
    SUE =  dif[0:(length_variances)] / variances
    return SUE
    # vecor of eps   
    
def f_CHTX(tes,total_assets,total_shares, qt = 4):
    # asset per share = total assets / total number of share 
    n =len(tes)
    CHTX =(  np.array(tes[0:(n-qt+1)]) - np.array(tes[(qt-1) : n])) / (np.array(total_assets[0:(n-qt+1)]) /np.array(total_shares[0:(n-qt+1)]))
    return CHTX

def f_growth(vector):
    n = len(vector)
    return ( vector[0:(n-1)] - vector[1:n] ) /  vector[1:n]

#%% path to the stocks folder : needs to be changed

stocks_dir = "C:\mDrive\Stanford\CS230 - Deep Learning 2024\Project\data\stocks"

#%% Parser

all_names = [name for name in os.listdir(stocks_dir) if os.path.isdir(os.path.join(stocks_dir, name))]
n_stocks = len(all_names)
dict_names = {name: {} for name in all_names}
stock_table = pd.DataFrame()
all_stocks_table = pd.DataFrame()
stock_table_fundamentals = pd.DataFrame()

file_list = ['balance-sheet-statement-quarterly','cash-flow-statement-quarterly','enterprise-values-quarterly',
              'enterprise-values-quarterly', 'key-metrics-quarterly']



# loop for all stocks or subset of stocks :
stocks_names = ['IBM','MICR'] # all_names
for name in stocks_names :
    stock_path = stocks_dir + "\\" + name
    stock_files = os.listdir(stock_path)
    dict_names[name] = {file_name.replace('-', '_'): pd.DataFrame() for file_name in stock_files}
    for file in file_list:
        file_path = stock_path + "\\" + file
        with open(file_path, 'r') as data_file:
            content = data_file.read()
            try:
                aux_table = json.loads(content) 
                aux_list =  aux_table[:len(aux_table)]
                aux_table = pd.DataFrame(aux_list)
                aux_columns = aux_table.columns
                stock_table[aux_columns] =aux_table
                # cleaning data : these columns are empty
                stock_table = stock_table.drop([ 'period','inventory',
                                  'taxAssets','deferredTaxLiabilitiesNonCurrent','link',
                                  'finalLink','deferredIncomeTax','accountsPayables',
                                  'salesMaturitiesOfInvestments','commonStockIssued','receivablesTurnover',
                                  'salesGeneralAndAdministrativeToRevenue', 'investedCapital'
                                  ], axis=1)
                
                # below is the construction of fundamental from raw params :
                    
                SYMBOL = stock_table['symbol']
                date = stock_table['date']
                filling_date = stock_table['fillingDate']
                acceptedDate = stock_table['acceptedDate']
                
                SUE = f_surprises(stock_table['earningsYield'])
                RSUP = f_surprises(stock_table['revenuePerShare'])
                CHTX = f_CHTX(stock_table['taxPayables'], stock_table['totalAssets'],stock_table['numberOfShares'])
                BM = np.array(stock_table['bookValuePerShare']) / np.array(stock_table['stockPrice'])
                PM = np.array(stock_table['netIncomePerShare']) / np.array(stock_table['revenuePerShare'])
                ROE = np.array(stock_table['netIncome']) / np.array(stock_table['totalStockholdersEquity'])
                Leverage = 1 / stock_table['debtToEquity']
                ATO = np.array(stock_table['revenuePerShare']) * np.array(stock_table['numberOfShares']) / np.array(stock_table['totalAssets'])
                DEPR = np.array(stock_table['depreciationAndAmortization']) / np.array(stock_table['tangibleAssetValue'])
                CASHDEBT = np.array(stock_table['operatingCashFlow']) / np.array(stock_table['totalDebt'])

                EP = np.array(stock_table['earningsYield'])
                CFP = np.array(stock_table['operatingCashFlowPerShare'])
                DY = np.array(stock_table['dividendYield'])
                RDM = np.array(stock_table['researchAndDdevelopementToRevenue'])
                RNA = np.array(stock_table['returnOnTangibleAssets'])
                CASH = np.array(stock_table['cashAtEndOfPeriod'])
                ACC =  np.array(stock_table['changeInWorkingCapital']) # suggestoion to directly take this variable 
                # growths
                GRLTNOA = f_growth(np.array(stock_table['totalAssets']))
                LGR = f_growth(np.array(stock_table['longTermDebt']))
                NINCR = f_growth(np.array(stock_table['earningsYield']))
                CHCSHO = f_growth(np.array(stock_table['numberOfShares']))
                CHPM = f_growth(PM)
                SGR = f_growth( np.array(stock_table['stockPrice']) / np.array(stock_table['priceToSalesRatio']) )


                NOA = np.array(stock_table['operatingCashFlow'])
                ROA = np.array(stock_table['returnOnTangibleAssets'])
                SP = np.array(stock_table['priceToSalesRatio'])
                TES = np.array(stock_table['taxPayables'])
                
                BM_IA = EP # This is false because this variable is not yet constructed




                fundamental_features_names = ['date','fillingDate','acceptedDate','symbol',
                    'SUE', 'RSUP', 'CHTX', 'BM', 'PM', 'ROE', 'Leverage', 'ATO', 'DEPR', 
                    'CASHDEBT', 'EP', 'CFP', 'DY', 'RDM', 'RNA', 'CASH', 'ACC', 'GRLTNOA', 
                    'LGR', 'NINCR', 'CHCSHO', 'CHPM', 'SGR', 'NOA', 'ROA', 'SP', 'TES', 'BM_IA'
                ]

                fundamental_features = [date,filling_date,acceptedDate , SYMBOL,
                    SUE, RSUP, CHTX, BM, PM, ROE, Leverage, ATO, DEPR, CASHDEBT, EP, CFP, DY, 
                    RDM, RNA, CASH, ACC, GRLTNOA, LGR, NINCR, CHCSHO, CHPM, SGR, NOA, ROA, SP, TES, BM_IA
                ]
                lengths = np.zeros(len(fundamental_features))
                for i in range(0,len(lengths)): lengths[i] = len(fundamental_features[i]) 
                min_length = int(min(lengths))
                for i in range(0,len(lengths)) : 
                    fundamental_features[i] = fundamental_features[i][0:min_length]
                stock_table_fundamentals = pd.DataFrame({fun_feat: array for fun_feat, array in zip(fundamental_features_names, fundamental_features)})
                # after collecting table of raw data and columns, below we construct the wanted financial features
            except:
                print(name,file,": this feature is not included in this parsing")
    all_stocks_table = pd.concat([all_stocks_table, stock_table_fundamentals], axis=0)            
#IBM = combined_df
all_stocks_table.to_parquet("two_stocks_fundamentals_example.parquet")
