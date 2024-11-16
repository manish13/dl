# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 00:15:17 2024

@author: zouit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from DL_functions_update import data_split
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct full paths to the data files inside the 'data' folder
data_dir = os.path.join(script_dir, "data")

realZ_sample_path = os.path.join(data_dir, "char.v1.txt")
realstock_return_path = os.path.join(data_dir, "ret.v1.txt")
realportfolio_return_path = os.path.join(data_dir, "realportfolio_return.txt")
realMKT_path = os.path.join(data_dir, "ff3.v1.txt")


# Load data : 
Z = np.tile(np.loadtxt(realZ_sample_path), (1, 30))
R1 = np.loadtxt(realstock_return_path)
R2 = np.loadtxt(realportfolio_return_path)
M = np.loadtxt(realMKT_path)
T = M.shape[0]  # number of periods




# Set parameters
split_ratio = 0.5 

# Split the data
z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, ff_n, port_n, t_train, n, p = data_split( Z, R1, M, R2, ratio=split_ratio, split='future'
)

# Function to compute moments
def compute_moments(data):
    # Flatten the data to 1D
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]  # Remove NaNs
    mean = np.mean(data_flat)
    variance = np.var(data_flat)
    skewness = skew(data_flat)
    kurt = kurtosis(data_flat)
    return mean, variance, skewness, kurt

# Compute moments for in-sample data
mean_r_train, var_r_train, skew_r_train, kurt_r_train = compute_moments(r_train)
mean_m_train, var_m_train, skew_m_train, kurt_m_train = compute_moments(m_train)


# Compute moments for out-of-sample data
mean_r_test, var_r_test, skew_r_test, kurt_r_test = compute_moments(r_test)
mean_m_test, var_m_test, skew_m_test, kurt_m_test = compute_moments(m_test)

# Create DataFrames to display the moments
moments_df = pd.DataFrame({
    'Moment': ['Mean', 'Variance', 'Skewness', 'Kurtosis'],
    'In-Sample': [mean_r_train, var_r_train, skew_r_train, kurt_r_train],
    'Out-of-Sample': [mean_r_test, var_r_test, skew_r_test, kurt_r_test]
})



moments_df2 = pd.DataFrame({
    'Moment': ['Mean', 'Variance', 'Skewness', 'Kurtosis'],
    'In-Sample': [mean_m_train, var_m_train, skew_m_train, kurt_m_train],
    'Out-of-Sample': [mean_m_test, var_m_test, skew_m_test, kurt_m_test]
})


print("Statistical Moments of returns:")
print(moments_df)
print("Statistical Moments of characs:")
print(moments_df2)


# Plot distributions
plt.figure(figsize=(14, 6))

# Histogram of returns for in-sample data
plt.subplot(1, 2, 1)
plt.hist(r_train.flatten(), bins=50, alpha=0.7, color='blue')
plt.title('In-Sample Return Distribution')
plt.xlabel('Return')
plt.ylabel('Frequency')

# Histogram of returns for out-of-sample data
plt.subplot(1, 2, 2)
plt.hist(r_test.flatten(), bins=50, alpha=0.7, color='orange')
plt.title('Out-of-Sample Return Distribution')
plt.xlabel('Return')
plt.ylabel('Frequency')


plt.figure(figsize=(14, 6))

# Histogram of returns for in-sample data
plt.subplot(1, 2, 1)
plt.hist(r_train.flatten(), bins=50, alpha=0.7, color='blue')
plt.title('In-Sample characteristics Distribution')
plt.xlabel('char')
plt.ylabel('Frequency')

# Histogram of returns for out-of-sample data
plt.subplot(1, 2, 2)
plt.hist(r_test.flatten(), bins=50, alpha=0.7, color='red')
plt.title('In-Sample characteristics Distribution')
plt.xlabel('Return')
plt.ylabel('Frequency')


plt.tight_layout()
plt.show()