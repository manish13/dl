from DL_functions_v2 import *
import numpy as np
import tensorflow as tf

# load data
ROOT = '/home/manish/code/dl/data/'


Z = np.loadtxt(f"{ROOT}char.v1.txt")
R1 = np.loadtxt(f"{ROOT}ret.v1.txt")
R2 = np.loadtxt(f"{ROOT}ret.v1.txt")
M = np.loadtxt(f"{ROOT}ff3.v1.txt")
T = M.shape[0] # number of periods
print(Z.shape, R1.shape, M.shape, T)
data_input = dict(characteristics=Z, stock_return=R1, target_return=R2, factor=M[:, 0:3])

# set parameters
training_para = dict(epoch=100, train_ratio=0.7, train_algo=tf.compat.v1.train.AdamOptimizer,
                     split="future", activation=tf.nn.tanh, start=1, batch_size=120, learning_rate=0.1,
                     Lambda1=0, Lambda2=0.1)
# design network layers
layer_size = [32, 16, 8, 4]

# construct deep factors
f, char, ltrain, lval, ltest = dl_alpha(data_input, layer_size, training_para)

import pickle
pickle.dump({'param':training_para, 'layers': layer_size}, open('data/parameters.pickle', 'wb'))
pickle.dump(f, open('data/factors.pickle', 'wb'))
pickle.dump(char, open('data/characteristics.pickle', 'wb'))
pickle.dump([ltrain, lval, ltest], open('data/loss.pickle', 'wb'))

