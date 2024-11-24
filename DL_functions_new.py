import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers, losses, metrics


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


def add_layer_1(inputs, in_size, out_size, activation, keep=0.5):
    '''
    add a neural layer on top of "inputs"
    :param inputs: lower layer
    :param in_size: size of lower layer (number of characteristics)
    :param out_size: size of new layer
    :param activation: activation function
    :param keep: dropout
    :return: new layer
    '''
    weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biases = tf.Variable(tf.random.normal([out_size]))
    wxb = tf.tensordot(tf.nn.dropout(inputs, keep),
                       weights, axes=[[2], [0]]) + biases
    outputs = activation(wxb)
    return outputs, weights, wxb


def get_batch(total, batch_number):
    '''
    create batches
    :param total: number of data points
    :param batch_number: number of batches
    :return: batches
    '''
    sample = np.arange(total)
    np.random.shuffle(sample)
    batch = np.array_split(sample, batch_number)
    return batch


def dc_loss(target, target_hat, weights_l1, l1, l2):
    loss1 = losses.MeanSquaredError()(target, target_hat)
    alpha  = tf.reduce_mean(target - target_hat,axis=0) 
    loss2 = losses.MeanSquaredError()(tf.zeros([alpha.shape[0]]), alpha)
    loss3 = weights_l1
    loss = loss1 + l1*loss2 + l2*loss3
    return loss

class DCModel(Model):
    def compile(self, optimizer, loss):
        super().compile(optimizer)
        self.loss = loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss_value = self.loss(y_pred[0], y_pred[1])
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss_value': loss_value}


def get_model(activation, inputShapes, layer_size, n):
    def sorting_module(x, e = 0.00001):
        mean, var = tf.nn.moments(x, axes=1, keepdims=True)
        zx = (x-mean)/(tf.sqrt(var)+e)
        a = -50*tf.exp(-5*zx)
        b = -50*tf.exp(5*zx)
        return tf.transpose(layers.Softmax(1)(a) - layers.Softmax(1)(b), perm=[0, 2, 1])
    
    def get_characteristic_layer(input, out_size, activation, keep=0.5):
        in_size = input.shape[-1]
        w = tf.Variable(tf.random.normal([in_size, out_size]))
        b = tf.Variable(tf.random.normal([out_size]))
        wTb = tf.tensordot(layers.Dropout(keep)(input), w, axes=[[2], [0]]) + b 
        return activation(wTb), w

    def deep_characteristics_module(x, L):
        w_l1 = tf.constant(0.0)
        for i in L:
            x, w = get_characteristic_layer(x, x.shape[-1], activation)
            w_l1 += tf.reduce_sum(tf.abs(w))
        x = sorting_module(x)
        return x, w_l1
    
    def deep_factor_returns_module(W, r):
        nobs = tf.shape(r)[0]
        Pd = tf.shape(W)[-1]
        r_tensor = tf.reshape(r, [nobs, n, 1])
        f_tensor = tf.matmul(W, r_tensor)
        return tf.reshape(f_tensor, [nobs, Pd])
    
    def beta_module(Z, out_n):
        Tn = Z.shape[1]
        b_d = tf.Variable(tf.random.normal([out_n[0], Tn]))
        b_b = tf.Variable(tf.random.normal([out_n[1], Tn]))
        return b_d, b_b
    
    Z = Input(shape=inputShapes[0], name='deep_characteristics')
    r = Input(shape=inputShapes[1], name='asset_returns')
    m = Input(shape=inputShapes[2], name='benchmark_factor_returns')

    W, w_l1 = deep_characteristics_module(Z, layer_size)
    f = deep_factor_returns_module(W, r)
    b_d, b_b = beta_module(Z, [layer_size[-1], inputShapes[2][0]])
    r_hat = tf.matmul(f, b_d) + tf.matmul(m, b_b)
    
    model = Model(inputs=[Z, r, m], outputs=[r_hat], name='deep_factor_model')
    
    return model, w_l1, r_hat, r
    
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
    model, w_l1, y_hat, y = get_model(param['activation'], inputShapes, layer_size, n)

    # print(model.summary())
    # tf.keras.utils.plot_model(model, 'deep_factor_model.png', show_shapes=True)

    model.compile(
        loss = dc_loss(y, y_hat, w_l1, param['Lambda1'], param['Lambda2']),
        optimizer=param['train_algo'](),
        # metrics= [metrics.MeanAbsolutePercentageError(), metrics.MeanSquaredError()],
    )

    history = model.fit([z_train, r_train, m_train], 
                        [target_train],
                        batch_size=param['batch_size'], 
                        epochs=param['epoch'], 
                        validation_split=0.3)
    
    
    test_scores = model.evaluate([z_test, r_test, m_test], [target_test], verbose=2)
    
    print('Test loss', test_scores[0])
    
    return model, history, test_scores
    
if __name__ == '__main__':
    import numpy as np

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
    model, history, test_scores = main(data_input, layer_size, training_para)

    # import pickle
    # pickle.dump({'param':training_para, 'layers': layer_size}, open('data/parameters.pickle', 'wb'))
    # pickle.dump(f, open('data/factors.pickle', 'wb'))
    # pickle.dump(char, open('data/characteristics.pickle', 'wb'))
    # pickle.dump([ltrain, lval, ltest], open('data/loss.pickle', 'wb'))
