'''Trains a simple binarize CNN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 99.110% test accuracy after 20 epochs using tensorflow backend
'''

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

from model import Binary_Net
from config import *

# [1]: 加载数据集并准备好数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, classes) * 2 - 1

# [1]: 加载网络结构和相关优化器,损失函数,回调函数
model = Binary_Net(kernel_size=kernel_size,img_rows=img_rows,img_cols=img_cols, channels=channels,
                data_format=data_format,H=H,kernel_lr_multiplier=kernel_lr_multiplier,use_bias=use_bias,
                epsilon=epsilon,momentum=momentum,classes=classes)

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
callbacks = [
    ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=1e-9, epsilon=0.01, verbose=1),
    EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1),
    ModelCheckpoint(monitor='val_acc',
                     filepath='logs/weights/Binary_Net_{epoch:02d}_{val_acc:.3f}.h5',
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto',
                     verbose=1,
                     period=1),
    lr_scheduler
            ]

# [3]: 训练并评估
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=callbacks)
score = model.evaluate(X_test, Y_test, verbose=2)
print('Test score:', score[0])
print('Test accuracy:', score[1])