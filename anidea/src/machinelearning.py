# AnIdea (c) 2019 Dominique F. Garmier All Rights Reserved
# Version: pre2.1
# --------------------------------------------------------

import numpy as np
import tensorflow as tf
from src.tfopsbatchjac import batch_jacobian


def broadcastVars(t, bc):

    x0_broadcasted = tf.broadcast_to(tf.transpose(tf.expand_dims(tf.expand_dims(bc[0,:], axis=-1), axis=-1), perm=(1,2,0)), (1, tf.shape(t)[1], tf.shape(bc)[1]))
    v0_broadcasted = tf.broadcast_to(tf.transpose(tf.expand_dims(tf.expand_dims(bc[1,:], axis=-1), axis=-1), perm=(1,2,0)), (1, tf.shape(t)[1], tf.shape(bc)[1]))

    t_broadcasted = tf.broadcast_to(t, ((tf.shape(t)[0], tf.shape(t)[1], tf.shape(bc)[1])))

    return x0_broadcasted, v0_broadcasted, t_broadcasted


def makeBatch(time_batch_size, time_range, bc_batch_size, bc_var):

    bs = int(np.sqrt(bc_batch_size))

    x0 = np.pi * ((2 * np.random.rand(bs))-1)
    v0 = bc_var * ((2 * np.random.rand(bs))-1)

    x0 = np.broadcast_to(x0, (bs, bs))
    v0 = np.transpose(np.broadcast_to(v0, (bs, bs)))

    bc = np.reshape([x0, v0], (2, bs*bs))
    t = np.reshape(np.linspace(0, time_range, time_batch_size), (1, -1, 1))
    return t, bc


def staticDenseLayer(id, l1_size, l0_size, bc, l0):
    '''Layer with Static Weights, returns layer as matrix'''

    width = tf.shape(bc)[1]
    
    std = 0.1
    w = tf.Variable(tf.random_normal([l1_size, l0_size, 1], stddev=std), name=('w' + str(id)))
    b = tf.Variable(tf.random_normal([l1_size, 1, 1], stddev=std), name=('w' + str(id)))

    w_ = tf.broadcast_to(w, ((tf.shape(w)[0], tf.shape(w)[1], width)))
    b_ = tf.broadcast_to(b, ((tf.shape(b)[0], tf.shape(b)[1], width)))
    
    l1 = tf.add(tf.einsum('ijk,jlk->ilk', w_, l0), b_)
    return l1


def dynamicNNDenseLayer(id, l1_size, l0_size, l0_, bc, nn_fact):
    ''' Layer with Dynamic Weights, returns layer as matrix.'''
    if type(nn_fact[0])==list:
        nn_fact = nn_fact[0]

    std = 0.01 # standard deviation for inital values of variables
    alpha = 0.01 # leaky relu alpha

    w_size = l1_size * l0_size
    b_size = l1_size * 1

    w_shape = (l1_size, l0_size, -1)
    b_shape = (l1_size, 1, -1)

    nw_layers = len(nn_fact)
    nb_layers = len(nn_fact)

    # contains the tf variables
    w_layers = []
    b_layers = []

    w_sizes = []
    w_sizes.append(2) # bc size
    
    b_sizes = []
    b_sizes.append(2) # bc size

    # read sizes of different layers form nn_fact
    for layer in range(nw_layers):
        w_sizes.append(int(nn_fact[layer] * w_size))

    for layer in range(nb_layers):
        b_sizes.append(int(nn_fact[layer] * b_size))

    # output layer size
    w_sizes.append(w_size)
    b_sizes.append(b_size)


    for layer in range(nw_layers + 1):

        if layer == 0:
            l0 = bc
        else:
            l0 = w_layers[layer - 1]

        w = tf.Variable(tf.random_normal([w_sizes[layer + 1], w_sizes[layer]], stddev=std), name=('ww' + str(id) + str(layer + 1)))
        b = tf.Variable(tf.random_normal([w_sizes[layer + 1], 1], stddev=std), name=('wb' + str(id) + str(layer + 1)))

        l_ = tf.add(tf.matmul(w, l0), b)

        if layer == nw_layers:
            l = tf.nn.leaky_relu(l_, alpha=alpha)
        else:
            l = l_

        w_layers.append(l)


    for layer in range(nb_layers + 1):

        if layer == 0:
            l0 = bc
        else:
            l0 = b_layers[layer - 1]

        w = tf.Variable(tf.random_normal([b_sizes[layer + 1], b_sizes[layer]], stddev=std), name=('bw' + str(id) + str(layer + 1)))
        b = tf.Variable(tf.random_normal([b_sizes[layer + 1], 1], stddev=std), name=('bb' + str(id) + str(layer + 1)))

        l_ = tf.add(tf.matmul(w, l0), b)

        if layer == nw_layers:
            l = tf.nn.leaky_relu(l_, alpha=alpha)
        else:
            l = l_

        b_layers.append(l)

    w = tf.reshape(w_layers[-1], w_shape)
    b = tf.reshape(b_layers[-1], b_shape)

    l1 = tf.add(tf.einsum('ijk,jlk->ilk', w, l0_), b)
    
    return l1


def dynamicPolyDenseLayer(id, l1_size, l0_size, l0_, bc, order):
    ''' Layer with dynamic weights, weights are polinomial functions of initial conditions, 
        returns layer as matrix.'''

    std = 0.01 # standard deviation for inital values of variables

    w_size = l1_size * l0_size
    b_size = l1_size * 1

    w_shape = (l1_size, l0_size, -1)
    b_shape = (l1_size, 1, -1)

    if order == 1:

        ww = tf.Variable(tf.random_normal([w_size, 2], stddev=std), name=(str(id) + '-poly-ww'))
        wb = tf.Variable(tf.random_normal([w_size, 1], stddev=std), name=(str(id) + '-poly-wb'))

        bw = tf.Variable(tf.random_normal([b_size, 2], stddev=std), name=(str(id) + '-poly-bw'))
        bb = tf.Variable(tf.random_normal([b_size, 1], stddev=std), name=(str(id) + '-poly-bb'))

        w = tf.reshape(tf.add(tf.matmul(ww, bc), wb), w_shape)
        b = tf.reshape(tf.add(tf.matmul(bw, bc), bb), b_shape)

        l1 = tf.add(tf.einsum('ijk,jlk->ilk', w, l0_), b)

        return l1

    elif order == 2:

        ww1 = tf.Variable(tf.random_normal([w_size, 2], stddev=std), name=(str(id) + '-poly-ww1'))
        ww2 = tf.Variable(tf.random_normal([w_size, 2], stddev=std), name=(str(id) + '-poly-ww2'))
        ww3 = tf.Variable(tf.random_normal([w_size, 1], stddev=std), name=(str(id) + '-poly-ww3'))
        wb = tf.Variable(tf.random_normal([w_size, 1], stddev=std), name=(str(id) + '-poly-wb'))

        bw1 = tf.Variable(tf.random_normal([b_size, 2], stddev=std), name=(str(id) + '-poly-bw1'))
        bw2 = tf.Variable(tf.random_normal([b_size, 2], stddev=std), name=(str(id) + '-poly-bw2'))
        bw3 = tf.Variable(tf.random_normal([b_size, 1], stddev=std), name=(str(id) + '-poly-bw3'))
        bb = tf.Variable(tf.random_normal([b_size, 1], stddev=std), name=(str(id) + '-poly-bb'))

        w = tf.reshape(tf.add(tf.add(tf.add(tf.matmul(ww2, tf.square(bc)), tf.matmul(ww1, bc)), tf.matmul(ww3, tf.reshape(tf.reduce_prod(bc, axis=0), (1, -1)))), wb), w_shape)
        b = tf.reshape(tf.add(tf.add(tf.add(tf.matmul(bw2, tf.square(bc)), tf.matmul(bw1, bc)), tf.matmul(bw3, tf.reshape(tf.reduce_prod(bc, axis=0), (1, -1)))), bb), b_shape)

        l1 = tf.add(tf.einsum('ijk,jlk->ilk', w, l0_), b)
        
        return l1

    else:
        raise AttributeError("poly dense layers only accept polinomials of degree n <= 2! ")


def buildNN(t_broadcasted, bc, model_architecture):
    ''' Create NN with architecture: model_architecture as a function of t_bdc and bc,
        returns the last layer aka output layer as matrix'''

    layers = []
    
    # appending l0 to layers (input layer)
    layers.append(t_broadcasted)

    # iterating through layers 1 ... n
    for i in range(len(model_architecture) - 1):
        layer_id = i + 1

        # check if layer is dynmaic or static
        if type(model_architecture[layer_id][1]) == list:
            new_layer = dynamicNNDenseLayer(layer_id, model_architecture[layer_id][0], model_architecture[layer_id - 1][0], layers[layer_id - 1], bc, model_architecture[layer_id][1])

        elif type(model_architecture[layer_id][1]) == str:
            order = 0
            if model_architecture[layer_id][1] == 'output layer':
                order = 2
            else:
                order = int(model_architecture[layer_id][1][1:])

            new_layer = dynamicPolyDenseLayer(layer_id, model_architecture[layer_id][0], model_architecture[layer_id - 1][0], layers[layer_id - 1], bc, order)

        else:
            new_layer = staticDenseLayer(layer_id, model_architecture[layer_id][0], model_architecture[layer_id - 1][0], bc, layers[layer_id - 1])

        # if last layer
        if not layer_id == (len(model_architecture) - 1):
            # between layer activation function
            layer = tf.tanh(new_layer)
        else:
            layer = new_layer

        # add calculated layer to layers
        layers.append(layer)

    return layers[len(layers) - 1]


def nnRectifier(N, x0_bdc, v0_bdc, t_bdc):
    '''rectifies the NN to satisfie x0_bdc and v0_bdc at t=0,
        returns N_r, dN_r, ddN_r'''

    _N = tf.reshape(N, (-1, 1))
    _t = tf.reshape(t_bdc, (-1, 1))

    _dN = tf.reshape(batch_jacobian(_N, _t),(-1, 1))
    _ddN = tf.reshape(batch_jacobian(_dN, _t),(-1, 1))

    dN = tf.reshape(_dN, tf.shape(N))
    ddN = tf.reshape(_ddN, tf.shape(N))

    # rectifier parameter function
    f = tf.square(t_bdc)
    df = 2 * t_bdc
    ddf = 2.

    # rectifier

    N_r = tf.multiply(N, f) + tf.multiply(v0_bdc, t_bdc) + x0_bdc
    dN_r = tf.multiply(N, df) + tf.multiply(dN, f) + v0_bdc
    ddN_r = tf.multiply(N, ddf) + tf.multiply(dN, df) + tf.multiply(dN, df) + tf.multiply(ddN, f)

    return N_r, dN_r, ddN_r


def lossFunction(N_r, dN_r, ddN_r, lossfunction, stepsize, omega):
    ''' computes the loss of NN, returns loss, show_loss, error
        
        can compute linloss aswell as pidloss spcified by lossfunction
        
        stepsize is only relevant for pid loss and refers to the time intervals in t_batch
        '''

    # differential equation
    error = tf.multiply(tf.sin(N_r), omega**2) + ddN_r

    # linear loss
    if lossfunction == 'linloss':

        loss = tf.reduce_mean(tf.square(error))

    # proportional, integral, differential loss 
    if lossfunction == 'pidloss':

        # pid constants 
        a = 1
        b = 1
        c = 1
        
        # proportional part
        P = tf.reduce_mean(tf.square(error))

        # integral part
        I = tf.reduce_mean(tf.square(tf.cumsum(error, axis=1))) / stepsize

        # differential part
        __ = tf.roll(error, 1, axis=1)
        _ = __[:,1:,:]
        e_rolled = tf.concat([_, tf.ones((tf.shape(error)[0],1,tf.shape(error)[2]))], axis=1)
        D = tf.reduce_mean(tf.square(error - e_rolled)) / stepsize

        loss = a*P + b*I + c*D

    show_loss = tf.reduce_mean(tf.square(error))
    return loss, show_loss, error


def buildModel(hyperparameters, omega):
    ''' builds model with hyperparameters,
        returns lr, t, bc, N_r, dN_r, ddN_r, loss, show_loss, error, optimizer'''


    # vars / placeholders
    h = hyperparameters['t_range'] / hyperparameters['t_batchsize']

    lr = tf.placeholder(dtype=tf.float32, shape=(), name='lr')
    t = tf.placeholder(dtype=tf.float32, shape=(1, None, 1), name='t')
    bc = tf.placeholder(dtype=tf.float32, shape=(2, None), name='bc') 
    
    x0_bdc, v0_bdc, t_bdc = broadcastVars(t, bc)
    
    # defining the NN itself
    N = buildNN(t_bdc, bc, hyperparameters['model_architecture'])

    # rectifier and lossfunction
    N_r, dN_r, ddN_r = nnRectifier(N, x0_bdc, v0_bdc, t_bdc)
    loss, show_loss, error = lossFunction(N_r, dN_r, ddN_r, lossfunction = hyperparameters['loss_function'], stepsize=h, omega=omega)

    # optimizer
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    # inits
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep = None)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    return lr, t, bc, N_r, dN_r, ddN_r, loss, show_loss, error, optimizer, init, saver, gpu_options
