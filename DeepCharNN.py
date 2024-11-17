import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers, losses, metrics
from tensorflow.keras.utils import plot_model


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

class SortingLayer(layers.Layer):
  def __init__(self, e):
    super(SortingLayer, self).__init__()
    self._name = 'SortL'
    self.e = e

  def build(self, input_shape):
    super(SortingLayer, self).build(input_shape)

  def call(self, inputs):
    mean, var = tf.nn.moments(inputs, axes=1, keepdims=True)
    zx = (inputs-mean)/(tf.sqrt(var)+self.e)
    a = -50*tf.exp(-5*zx)
    b = -50*tf.exp(5*zx)
    return tf.transpose(layers.Softmax(1)(a) - layers.Softmax(1)(b), perm=[0, 2, 1])
  
class CharLayer(layers.Layer):
  def __init__(self, num_outputs, activation, lambda2, idx, keep=0.5):
    super(CharLayer, self).__init__()
    self._name = f'CharL{idx}'
    self.num_outputs = num_outputs
    self.activation = activation 
    self.lambda2 = lambda2
    self.keep = keep

  def build(self, input_shape):
    self.w = self.add_weight(name="w", shape=(input_shape[-1], self.num_outputs), initializer='random_normal', trainable=True)
    self.b = self.add_weight(name="b", shape=(self.num_outputs,), initializer='zeros', trainable=True)
    super(CharLayer, self).build(input_shape)

  def call(self, input):
    dol = layers.Dropout(self.keep)(input)
    wTx = tf.tensordot(dol, self.w, axes=[[2],[0]]) + self.b
    self.add_loss(tf.reduce_sum(tf.abs(self.w))*self.lambda2)
    return self.activation(wTx)

def make_deep_char_network(x, layer_size, activation, lambda2, e=0.00001):
    lsize = [x.shape[-1]]+layer_size
    for i,l in enumerate(layer_size):
        x = CharLayer(lsize[i+1], activation, lambda2, i)(x)
    x = SortingLayer(e)(x)
    return x

class DeepFactorLayer(layers.Layer):
    def __init__(self, output_size):
        super(DeepFactorLayer, self).__init__()
        self._name = 'DFacL'
        self.n = output_size

    def build(self, input_shape):
        super(DeepFactorLayer, self).build(input_shape)

    def call(self, inputs):
        W, r = inputs
        nobs = tf.shape(r)[0]
        Pd = tf.shape(W)[1]
        r_tensor = tf.reshape(r, [nobs, self.n, 1])
        f_tensor = tf.matmul(W, r_tensor)
        return tf.reshape(f_tensor, [nobs, Pd])

class DeepBetaLayer(layers.Layer):
    def __init__(self, output_size, obs_size):
        super(DeepBetaLayer, self).__init__()
        self._name = 'BetaL'
        self.out_n = output_size
        self.Tn = obs_size

    def build(self, input_shape):
        self.b_d = self.add_weight('b_d', shape=[self.out_n[0], self.Tn], trainable=True)
        self.b_b = self.add_weight('b_b', shape=[self.out_n[1], self.Tn], trainable=True)
        super(DeepBetaLayer, self).build(input_shape)

    def call(self, input):   
        return self.b_d, self.b_b

class DeepFactorReturnsLayer(layers.Layer):
    def __init__(self, lambda1):
        super(DeepFactorReturnsLayer, self).__init__()
        self._name = 'FretL'
        self.lambda1 = lambda1

    def build(self, input_shape):
        super(DeepFactorReturnsLayer, self).build(input_shape)

    def call(self, inputs): 
       f, m, b_d, b_b, r = inputs
       r_hat = tf.matmul(f, b_d) + tf.matmul(m, b_b)
       alpha  = tf.reduce_mean(r - r_hat,axis=0) 
       self.add_loss(tf.reduce_mean(tf.square(alpha))*self.lambda1)
       return r_hat
   
def get_model(activation, inputShapes, layer_size, n, lambda1, lambda2):
    Z = Input(shape=inputShapes[0], name='raw_chars')
    r = Input(shape=inputShapes[1], name='asset_ret')
    m = Input(shape=inputShapes[2], name='bfac_ret')

    W = make_deep_char_network(Z, layer_size, activation, lambda2)
    f = DeepFactorLayer(n)([W, r])
    b_d, b_b= DeepBetaLayer([layer_size[-1], inputShapes[2][0]], Z.shape[1])(W)

    r_hat = DeepFactorReturnsLayer(lambda1)([f, m, b_d, b_b, r])
    
    model = Model(inputs=[Z, r, m], outputs=r_hat, name='deep_factor_model')
    
    return model
    
def main(data, layer_size, param):
    print(tf.__version__)
    z_train, r_train, m_train, target_train, z_test, r_test, m_test, target_test, n = \
        data_split(data['characteristics'], 
                   data['stock_return'], 
                   data['factor'], 
                   data['target_return'],
                   param['train_ratio'],
                   param['split'])
    
    inputShapes = [z_train.shape[1:], r_train.shape[1:], m_train.shape[1:]]
    model = get_model(param['activation'], inputShapes, layer_size, n, param['Lambda1'], param['Lambda2'])

    model.compile(
        loss = losses.MeanSquaredError(),
        optimizer = param['train_algo'](),
        metrics = metrics.MeanSquaredError(),
    )
    print(model.summary())
    # plot_model(model, 'data/deep_factor_model.png', show_shapes=True)

    history = model.fit([z_train, r_train, m_train], 
                        [target_train],
                        batch_size=param['batch_size'], 
                        epochs=param['epoch'], 
                        validation_split=0.3)
    
    
    test_scores = model.evaluate([z_test, r_test, m_test], [target_test], verbose=2)
    
    return model, history, test_scores
    
if __name__ == '__main__':
    import numpy as np

    # load data
    ROOT = '/home/manish/code/dl/data/'

    Z = np.loadtxt(f"{ROOT}char.v1.txt").astype(np.float32)
    R1 = np.loadtxt(f"{ROOT}ret.v1.txt").astype(np.float32)
    R2 = np.loadtxt(f"{ROOT}ret.v1.txt").astype(np.float32)
    M = np.loadtxt(f"{ROOT}ff3.v1.txt").astype(np.float32)
    T = M.shape[0] # number of periods
    print(Z.shape, R1.shape, M.shape, T)

    data_input = dict(characteristics=Z, stock_return=R1, target_return=R2, factor=M[:, 0:3])

    # set parameters
    training_para = dict(epoch=200, train_ratio=0.7, train_algo=tf.compat.v1.train.AdamOptimizer,
                        split="future", activation=tf.nn.tanh, start=1, batch_size=120, learning_rate=0.1,
                        Lambda1=0, Lambda2=0.1)

    # design network layers
    layer_size = [32, 16, 8, 4]

    # construct deep factors
    model, history, test_scores = main(data_input, layer_size, training_para)

    import pickle
    pickle.dump({'param':training_para, 'layers': layer_size}, open('data/parameters.pickle', 'wb'))
    # pickle.dump(f, open('data/factors.pickle', 'wb'))
    # pickle.dump(char, open('data/characteristics.pickle', 'wb'))
    pickle.dump([history.history['loss'], history.history['val_loss'], history.history['mean_squared_error'], history.history['val_mean_squared_error']], open('data/loss.pickle', 'wb'))
