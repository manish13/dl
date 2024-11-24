import pandas as pd, numpy as np, datetime as dt
import os

from characteristics import input_map
from fundamental_factors import base_fundamental_data_location
from characteristics import CharacterisitcMaker

ROOT = '/home/manish/code/dl/data/standardized_factors/'

# create universe


# create returns


# create various factor returns


# create characteristic data
def file2name(ROOT):
    mp = {}
    for i in os.listdir(ROOT):
        mp[i] = i.split('.')[0].upper().replace('_', '')
    return mp