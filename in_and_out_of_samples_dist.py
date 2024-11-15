# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 00:15:17 2024

@author: zouit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from DL_functions_v2 import data_split


# Load data
Z = np.tile(np.loadtxt("realZ_sample.txt"), (1, 30))
R1 = np.loadtxt("realstock_return.txt")
R2 = np.loadtxt("realportfolio_return.txt")
M = np.loadtxt("realMKT.txt")
T = M.shape[0]  # number of periods



# Set parameters
split_ratio = 0.8  # 80% training, 20% testing

# Split the data
z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test = data_split(
    Z, R1, M, R2, ratio=split_ratio, split='future'
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

print("Statistical Moments:")
print(moments_df)

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

plt.tight_layout()
plt.show()