import numpy as np, pandas as pd, pickle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input, layers, losses, metrics, regularizers, optimizers
from tensorflow.keras.utils import plot_model
tf.random.set_seed(420)

def data_split(z_data, r_data, m_data, target_data, ratio, split):
    '''
    split data
    :param z_data: characteristics
    :param r_data: stock return
    :param m_data: benchmark model
    :param target_data: target portfolio
    :param ratio: train/test ratio for split
    :param split: if "future", split data into past/future periods using "ratio",
                  if integer "t", select test data every t periods
    :return: train and test data
    '''

    #ff_n = m_data.shape[1]  # factor number
    #port_n = target_data.shape[1]  # target (portfolio) number
    [t, n] = r_data.shape  # time length and stock number
    p = int(z_data.shape[1] / n)  # characteristics number
    z_data = z_data.reshape((t, p, n)).transpose((0, 2, 1))   # dim: (t,n,p)

    # train sample and test sample
    if split == 'future':
        test_idx = np.arange(int(t * ratio), t)
    else:
        test_idx = np.arange(0, t, split)

    train_idx = np.setdiff1d(np.arange(t), test_idx)
    #t_train = len(train_idx)
    z_train = z_data[train_idx]
    z_test = z_data[test_idx]
    r_train = r_data[train_idx]
    r_test = r_data[test_idx]
    target_train = target_data[train_idx]
    target_test = target_data[test_idx]
    m_train = m_data[train_idx]
    m_test = m_data[test_idx]

    return z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, n

class CharLayer(layers.Layer):
  def __init__(self, num_outputs, activation, lambda2, idx, keep=0.5):
    super(CharLayer, self).__init__()
    self._name = f'CharL{idx}'
    self.num_outputs = num_outputs
    self.activation = activation 
    self.lambda2 = lambda2
    self.keep = keep
    self.dropout = layers.Dropout(self.keep)  # Create Dropout layer in __init__

  def build(self, input_shape):
    initializer= tf.keras.initializers.GlorotNormal(seed=200)
    self.w = self.add_weight(name="w", shape=(input_shape[-1], self.num_outputs), initializer=initializer, trainable=True)
    self.b = self.add_weight(name="b", shape=(self.num_outputs,), initializer='zeros', trainable=True)
    super(CharLayer, self).build(input_shape)

  def call(self, input):
    #dol = layers.Dropout(self.keep)(input)
    dol = self.dropout(input)  # Use the Dropout layer created in __init__
    wTx = tf.tensordot(dol, self.w, axes=[[2],[0]]) + self.b  # 2 is charcteristic and 0 is time axis
    # the below is L1 regularization.
    # to make it L2, uncomment the below line
    # self.add_loss(tf.reduce_sum(tf.square(self.w)) * self.lambda2)
    self.add_loss(tf.reduce_sum(tf.abs(self.w))*self.lambda2)
    return self.activation(wTx)

class SortingLayer(layers.Layer):
  # this layer replicates "human" sorting of characteristics to create factors
  def __init__(self, e):
    super(SortingLayer, self).__init__()
    self._name = 'SortL'
    self.e = e

  def build(self, input_shape):
    super(SortingLayer, self).build(input_shape)

  def call(self, inputs):
    mean, var = tf.nn.moments(inputs, axes=1, keepdims=True)
    zx = (inputs-mean)/(tf.sqrt(var)+self.e)
    # print (f'zx shape: {zx.shape}') --> (None, 2941, 2)
    a = -50*tf.exp(-5*zx)
    b = -50*tf.exp(5*zx)
    # 0 = time steps, 1 = num stocks, 2 = num characteristics
    # original input, num characteristics = 50
    # before sorting layer, num characteristics reduced to = 2
    return tf.transpose(layers.Softmax(1)(a) - layers.Softmax(1)(b), perm=[0, 2, 1]) # dimension of return is (None, 2, 2941)

class DeepFactorLayer(layers.Layer):
    # used to get deep factor returns
    def __init__(self, output_size):
        super(DeepFactorLayer, self).__init__()
        self._name = 'DFacL'
        self.n = output_size

    def build(self, input_shape):
        super(DeepFactorLayer, self).build(input_shape)

    def call(self, inputs):
        W, r = inputs
        nobs = tf.shape(r)[0] # number of observations (time stamps)
        Pd = tf.shape(W)[1] # number of deep characteristics
        r_tensor = tf.reshape(r, [nobs, self.n, 1])
        # print (f'W shape: {W.shape}, r shape: {r.shape}, r_tensor shape: {r_tensor.shape}')
        f_tensor = tf.matmul(W, r_tensor) # (None, 2, 2941) x (None, 2941, 1) = (None, 2, 1)
        return tf.reshape(f_tensor, [nobs, Pd])

class BetaLayer(layers.Layer):
    def __init__(self, output_size, obs_size, name):
        super(BetaLayer, self).__init__()
        self._name = f'{name}BetaL'
        self.out_n = output_size
        self.Tn = obs_size

    def build(self, input_shape):
        self.beta = self.add_weight('b_d', shape=[self.out_n, self.Tn], trainable=True)
        super(BetaLayer, self).build(input_shape)

    def call(self, fac): # fac is return of benchmark factors.  
        return tf.matmul(fac, self.beta)  # linear regression

class DeepFactorReturnsLayer(layers.Layer):
    def __init__(self, lambda1):
        super(DeepFactorReturnsLayer, self).__init__()
        self._name = 'FretL'
        self.lambda1 = lambda1

    def build(self, input_shape):
        super(DeepFactorReturnsLayer, self).build(input_shape)

    def call(self, inputs): 
       B_dxf_d, B_bxf_b, r = inputs
       r_hat = B_dxf_d + B_bxf_b
       alpha  = tf.reduce_mean(r - r_hat,axis=0) 
       self.add_loss(tf.reduce_mean(tf.square(alpha))*self.lambda1) # makes no sense to multiply by lambda1
       return r_hat

def make_deep_beta_network(x, fac, layer_size, activation, lambda3, suffix):
    lsize = [x.shape[-1]] + layer_size
    # a, b, c = x.shape
    # x = tf.reshape(x, (tf.shape(x)[0], b*c))
    for i, l in enumerate(layer_size):
        x = layers.Dense(lsize[i+1], activation=activation, kernel_regularizer=regularizers.l2(lambda3), name=f'{suffix}.{i}')(x)
    b_d = tf.transpose(x, perm=[0, 2, 1])
    return tf.matmul(fac, b_d)

def make_deep_char_network(x, layer_size, activation, lambda2, e=0.00001):
    lsize = [x.shape[-1]] + layer_size # [50] + [32, 16, 8, 2], where 50 is the number of characteristics (input dimension)
    for i,l in enumerate(layer_size):
        # num_outputs, activation, lambda2, idx
        x = CharLayer(lsize[i+1], activation, lambda2, f'_C.{i}')(x)
    x = SortingLayer(e)(x)
    return x

def get_model(activation, inputShapes, char_layer_size, beta_layer_size, bfac_layer_size, n, lambda_list):
    lambda1, lambda2, lambda3 = lambda_list
    Z = Input(shape=inputShapes[0], name='raw_chars')
    r = Input(shape=inputShapes[1], name='asset_ret')
    m = Input(shape=inputShapes[2], name='bfac_ret')
    u = Input(shape=inputShapes[1], name='mask')

    W = make_deep_char_network(Z, char_layer_size, activation, lambda2)
    f = DeepFactorLayer(n)([W, r])

    B_dxf_d = make_deep_beta_network(Z, f, beta_layer_size, activation, lambda3, 'D')
    
    # B_bxf_b = BetaLayer(inputShapes[2][0], Z.shape[1], 'benchmark')(m)
    B_bxf_b = make_deep_beta_network(Z, m, bfac_layer_size, activation, lambda3, 'B')
    
    r_hat = DeepFactorReturnsLayer(lambda1)([B_dxf_d, B_bxf_b, r])
    
    model = Model(inputs=[Z, r, m], outputs=r_hat, name='deep_factor_model')
    
    return model
    
def main(data, param):
    print("TensorFlow version = {}",tf.__version__)
    char_layer_size, beta_layer_size, bfac_layer_size = param['layers']
    assert char_layer_size[-1] == beta_layer_size[-1]

    print("Input Data shape: ", data['characteristics'].shape, data['stock_return'].shape, data['factor'].shape, data['target_return'].shape)
    # (341, 147050) (341, 2941) (341, 3) (341, 2941)
    
    z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, n = \
        data_split(data['characteristics'], 
                   data['stock_return'], 
                   data['factor'], 
                   data['target_return'],
                   param['train_ratio'],
                   param['split'])
    
    print("Train data shape:", z_train.shape, r_train.shape, m_train.shape, target_train.shape)
    # (238, 2941, 50) (238, 2941) (238, 3) (238, 2941)
    
    inputShapes = [z_train.shape[1:], r_train.shape[1:], m_train.shape[1:]]
    model = get_model(param['activation'], inputShapes, char_layer_size, beta_layer_size, bfac_layer_size,
                      n, [param['Lambda1'], param['Lambda2'], param['Lambda3']])

    model.compile(
        loss = losses.MeanSquaredError(),
        optimizer = param['train_algo'](learning_rate=param['learning_rate']),
        # optimizer = optimizers.RMSprop(learning_rate=param['learning_rate']),
        metrics = [metrics.MeanSquaredError()],
    )
    print(model.summary())
    # plot_model(model, 'data/deep_factor_model.png', show_shapes=True)

    history = model.fit([z_train, r_train, m_train], 
                        [target_train],
                        batch_size=param['batch_size'], 
                        epochs=param['epoch'],
                        # callbacks=[callbacks.EarlyStopping(patience=2)], 
                        validation_split=0.3)
    
    
    test_scores = model.evaluate([z_test, r_test, m_test], [target_test], verbose=2)
    
    return model, history, test_scores
    
if __name__ == '__main__':
    import numpy as np, pandas as pd

    # load data
    ROOT = './data/'

    # Z = np.loadtxt(f"{ROOT}char.v2.txt").astype(np.float32)
    # Z = np.concatenate([Z, (Z!=0).astype(float)], axis=1)
    # # Z =  np.random.randn(*Z.shape)
    # R1 = np.loadtxt(f"{ROOT}ret.v2.txt").astype(np.float32)
    # # R1 = np.random.randn(*R1.shape)
    # R2 = np.loadtxt(f"{ROOT}ret.v2.txt").astype(np.float32)
    # # R2 = np.random.randn(*R2.shape)
    # M = np.loadtxt(f"{ROOT}ff3.v1.txt").astype(np.float32)
    # # M = np.random.randn(*M.shape)
    # T = M.shape[0] # number of periods
    # print(Z.shape, R1.shape, M.shape, T)

    # data_input = dict(characteristics=Z, stock_return=R1, target_return=R2, factor=M[:, 0:3])

    # # set parameters
    # training_para = dict(epoch=100, train_ratio=0.7, train_algo=tf.compat.v1.train.AdamOptimizer,
    #                     split="future", activation=tf.nn.tanh, start=1, batch_size=120, 
    #                     learning_rate=.01, Lambda1=0.000, Lambda2=0.0001, Lambda3=0)

    # # design network layers
    # char_layer_size = [1]#[32, 16, 8, 2]
    # beta_layer_size = [1]#[64, 16, 2]  # default [4]

    # training_para = dict(epoch=100, train_ratio=0.7, train_algo=optimizers.AdamOptimizer,
    #                     split="future", activation=tf.nn.tanh, start=1, batch_size=50, 
    #                     learning_rate=.01, Lambda1=0.000, Lambda2=0.0001, Lambda3=0)
    # char_layer_size = [32, 16, 8, 2]
    # beta_layer_size = [8, 4, 2]

    Z = np.loadtxt(f"{ROOT}standardized_factors_MERGED.txt").astype(np.float32)
    
    Z = Z[:-1, :] #lag the characteristics by 1 period
    R1 = np.loadtxt(f"{ROOT}ret.v2.txt").astype(np.float32)[1:] #done to ensure same dimension as Z
    R1 = np.clip(R1, -0.2, .2)
    R2 = R1
    M = np.loadtxt(f"{ROOT}ff3.v1.txt").astype(np.float32)[1:] #done to ensure same dimension as Z
    U = pd.read_parquet(f'{ROOT}universe_monthly.parquet').values[1:] #done to ensure same dimension as Z
    # T = M.shape[0] # number of periods
    T, F = M.shape # number of periods
    print(Z.shape, R1.shape, M.shape, U.shape, T)

    data_input = dict(characteristics=Z, stock_return=R1, target_return=R2, factor=M[:, 0:3], mask=U)

    char_layer_size = [32, 16, 8, 2]
    beta_layer_size = [8, 4, 2]
    bfac_layer_size = [8, 4, F]

    training_para = dict(epoch=30, train_ratio=0.7, train_algo=optimizers.Adam,
                        split="future", activation=tf.nn.tanh, start=1, batch_size=75,
                        layers=[char_layer_size, beta_layer_size, bfac_layer_size], 
                        learning_rate=1e-3, Lambda1=1e-4, Lambda2=1e-5, Lambda3=1e-6) # l1: alpha, l2: char loading, l3: beta

    # construct deep factors
    model, history, test_scores = main(data_input, training_para)

    pickle.dump({'param':training_para}, open(f'{ROOT}/parameters.pickle', 'wb'))
    # pickle.dump(f, open('data/factors.pickle', 'wb'))
    # pickle.dump(char, open('data/characteristics.pickle', 'wb'))
    print('test_scores',test_scores)
    print(history)
    # pickle.dump(history, open(f'{ROOT}/history.pickle', 'wb'))
    pickle.dump([history.history['loss'], history.history['val_loss'], history.history['mean_squared_error'], history.history['val_mean_squared_error']], open(f'{ROOT}/loss.pickle', 'wb'))
