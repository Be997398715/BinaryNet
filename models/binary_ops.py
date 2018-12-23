# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)    # 使用tf.round函数； 功能：四舍五入,即0.9取1，0.1取0
    return x + K.stop_gradient(rounded - x)     # 使用tf.stop_gradient对反向传播进行截断


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    x = (0.5 * x) + 0.5     # 模拟sig_moid函数
    return K.clip(x, 0, 1)      # 按照论文中所说将激活函数的值限制在+1或-1,又因为sigmoid函数的值是在[0,1]区间的, 所以这里取min_value=0,max_value=1


def binary_sigmoid(x):
    '''Binary hard sigmoid for training binarized neural network.
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    '''Binary hard sigmoid for training binarized neural network.
     The neurons' activations binarization function
     It behaves like the sign function during forward propagation
     And like:
        hard_tanh(x) = 2 * _hard_sigmoid(x) - 1 
        clear gradient when |x| > 1 during back propagation
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    return 2 * round_through(_hard_sigmoid(x)) - 1      # 将tanh函数近似为2*sigmoid-1


def binarize(W, H=1):
    '''The weights' binarization function, 
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    # [-H, H] -> -H or H
    Wb = H * binary_tanh(W / H)
    return Wb


def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = binarize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb