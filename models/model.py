
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D

def binary_tanh(x):
    return binary_tanh_op(x)


def Binary_Net(kernel_size,img_rows,img_cols, channels,data_format,H,kernel_lr_multiplier,use_bias,epsilon,momentum,classes,pool_size):
	model = Sequential()
	# conv1
	model.add(BinaryConv2D(128, kernel_size=kernel_size, input_shape=(img_rows, img_cols, channels),
	                       data_format=data_format,
	                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
	                       padding='same', use_bias=use_bias, name='conv1'))
	model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=-1, name='bn1'))
	model.add(Activation(binary_tanh, name='act1'))
	# conv2
	model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
	                       data_format=data_format,
	                       padding='same', use_bias=use_bias, name='conv2'))
	model.add(MaxPooling2D(pool_size=pool_size, name='pool2', data_format=data_format))
	model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=-1, name='bn2'))
	model.add(Activation(binary_tanh, name='act2'))
	# conv3
	model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
	                       data_format=data_format,
	                       padding='same', use_bias=use_bias, name='conv3'))
	model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=-1, name='bn3'))
	model.add(Activation(binary_tanh, name='act3'))
	# conv4
	model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
	                       data_format=data_format,
	                       padding='same', use_bias=use_bias, name='conv4'))
	model.add(MaxPooling2D(pool_size=pool_size, name='pool4', data_format=data_format))
	model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=-1, name='bn4'))
	model.add(Activation(binary_tanh, name='act4'))
	model.add(Flatten())
	# dense1
	model.add(BinaryDense(1024, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
	model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
	model.add(Activation(binary_tanh, name='act5'))
	# dense2
	model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
	model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))
	return model
