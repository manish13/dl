# copy right @ Feng, Polson, and Xu "Deep Learning in Characteristics-Sorted Factor Models" (2019)

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

    Pb = m_data.shape[1]  # factor number
    Tn = target_data.shape[1]  # target (portfolio) number
    [t, n] = r_data.shape  # time length and stock number
    K0 = int(z_data.shape[1] / n)  # characteristics number
    z_data = z_data.reshape((t, K0, n)).transpose((0, 2, 1))   # dim: (t,n,K0)
    
    # test sample
    test_idx = np.arange(int(t * ratio), t)
    # train sample and val sample
    if split == 'future':
        val_idx = np.arange(int(t * ratio*ratio), int(t * ratio))
    else:
        val_idx = np.arange(0, t, split)

    train_idx = np.setdiff1d(np.setdiff1d(np.arange(t), test_idx), val_idx)
    t_train = len(train_idx)

    z_train = z_data[train_idx]
    z_val = z_data[val_idx]
    z_test = z_data[test_idx]

    r_train = r_data[train_idx]
    r_val = r_data[val_idx]
    r_test = r_data[test_idx]
    
    target_train = target_data[train_idx]
    target_val = target_data[val_idx]
    target_test = target_data[test_idx]

    m_train = m_data[train_idx]
    m_val = m_data[val_idx]
    m_test = m_data[test_idx]

    return z_train, r_train, m_train, target_train, z_val, r_val, m_val, target_val, z_test, r_test, m_test, target_test, Pb, Tn, t_train, n, K0


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


def dl_alpha(data, layer_size, para):
    '''
    construct deep factors
    :param data: a dict of input data
    :param layer_size: a list of neural layer sizes (from bottom to top)
    :param para: training and tuning parameters
    :return: constructed deep factors and deep characteristics
    '''
    print(tf.__version__)
    # split data to training sample and test sample 
    z_train, r_train, m_train, target_train, z_val, r_val, m_val, target_val, z_test, r_test, m_test, target_test, Pb, Tn, t_train, n, K0 = \
        data_split(data['characteristics'], data['stock_return'], data['factor'], data['target_return'],
                   para['train_ratio'], para['split'])
    
    assert z_train.shape[0] == r_train.shape[0]
    assert z_train.shape[0] == target_train.shape[0]
    assert r_train.shape[0] == t_train

    Pd = layer_size[-1]   # number of deep factors 

    # inputs
    z = tf.keras.Input(name='z', shape=(n, K0), dtype=tf.dtypes.float32)
    r = tf.keras.Input(name='r', shape=(None,), dtype=tf.dtypes.float32)
    m = tf.keras.Input(name='m', shape=(Pb,), dtype=tf.dtypes.float32)
    target = tf.keras.Input(name='target', shape=(Tn,), dtype=tf.dtypes.float32)

    # create graph for sorting
    with tf.compat.v1.name_scope('sorting_network'):
        # add 1st network (prior to sorting)
        L = len(layer_size)
        layer_size = np.insert(layer_size, 0, K0)
        layers_1 = [z]
        weights_l1 = tf.constant(0.0)
        for i in range(L):
            new_layer, weights, wxb = add_layer_1(
                layers_1[i], layer_size[i], layer_size[i + 1], para['activation'])
            layers_1.append(new_layer)
            if i < L -1:
                weights_l1 += (tf.reduce_sum(tf.abs(weights))) #- tf.reduce_sum(tf.abs(tf.linalg.diag_part(weights))))

        # softmax for factorweight
        mean, var = tf.nn.moments(layers_1[-1],axes=1,keepdims=True)
        normalized_char = (layers_1[-1] - mean)/(tf.sqrt(var)+0.00001)
        transformed_char_a = -50*tf.exp(-5*normalized_char)
        transformed_char_b = -50*tf.exp(5*normalized_char)
        w_tilde = tf.transpose(a=tf.nn.softmax(transformed_char_a, axis=1) - tf.nn.softmax(transformed_char_b, axis=1), perm=[0,2,1])

        # construct factors
        nobs = tf.shape(r)[0]
        r_tensor = tf.reshape(r, [nobs, n, 1])
        f_tensor = tf.matmul(w_tilde, r_tensor)
        f = tf.reshape(f_tensor, [nobs, Pd])

        # forecast return and alpha
        beta = tf.Variable(tf.random.normal([layer_size[-1], Tn])) 
        gamma = tf.Variable(tf.random.normal([Pb, Tn])) 
        target_hat = tf.matmul(f, beta) + tf.matmul(m, gamma)
        alpha  = tf.reduce_mean(target - target_hat,axis=0) 

        # define loss and training parameters
        zero = tf.zeros([Tn,])
        loss1 = tf.compat.v1.losses.mean_squared_error(target, target_hat)
        loss2 = tf.compat.v1.losses.mean_squared_error(zero, alpha)
        loss3 = weights_l1
        loss = loss1 + para['Lambda1']*loss2 + para['Lambda2']*loss3
        obj = para['train_algo'](para['learning_rate']).minimize(loss)

    batch_number = int(t_train / para['batch_size'])
    loss_train, loss_val, loss_test = [], [], []
    early_stopping = 10
    thresh = 0.000005
    stop_flag = 0

    # SGD training
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # train sorting network
        for i in range(para['epoch']):
            batch = get_batch(t_train, batch_number)

            for idx in range(batch_number):
                _ = sess.run(obj, feed_dict={
                    z: z_train[batch[idx]], r: r_train[batch[idx]], target: target_train[batch[idx]],
                    m: m_train[batch[idx]]})

            #train loss
            loss_train_, a, b, c = sess.run([loss, loss1, loss2, loss3], feed_dict={z: z_train, r: r_train, target: target_train, m: m_train})
            loss_train.append(loss_train_)

            #train loss
            loss_val_ = sess.run([loss], feed_dict={z: z_val, r: r_val, target: target_val, m: m_val})
            loss_val.append(loss_val_[0])

            #test loss
            loss_test_= sess.run( [loss], feed_dict={z: z_test, r: r_test, target: target_test, m: m_test})
            loss_test.append(loss_test_[0])
            
            print(f"epoch:{i} train:{loss_train_}(={a}+{b}+{c}), val:{loss_val_[0]}, test:{loss_test_[0]}")

            if np.isnan(loss_train_):
                break

            if i > 0:
                if loss_train[i-1] - loss_train[i] < thresh:
                    stop_flag += 1
                else:
                    stop_flag = 0
                if stop_flag >= early_stopping:
                    print('Early stopping at epoch:', i)
                    break


        # save constructed sort factors
        factor_in = sess.run(
            f, feed_dict={z: z_train, r: r_train, target: target_train, m: m_train})
        factor_out = sess.run(
            f, feed_dict={z: z_test, r: r_test, target: target_test, m: m_test})

        # characteristics

        deep_char = sess.run(layers_1[-1], feed_dict={z: np.concatenate((z_train,z_test),axis=0)})
        

    factor = np.concatenate((factor_in,factor_out),axis=0)
    nt, nnn, pp = deep_char.shape
    deep_char = deep_char.reshape(nt, nnn*pp) # todo check if this is right way to reshape
    return factor, deep_char, loss_train, loss_val, loss_test


if __name__ == '__main__':
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
