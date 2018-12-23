import h5py
import numpy as np
import time
import sys  
sys.path.append('./models')
import cv2

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

from config import *
from model import Binary_Net

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


# [2]: 修改训练好的权重文件, 并加载网络
def SignNumpy(x):
	return np.float16(2.*np.greater_equal(x,0)-1.)	# 将值变为+1或-1

weights_path = 'logs/weights/Binary_Net_21_0.99.h5'	
f = h5py.File(weights_path)
weigths_lists = list(f.keys())		# ['act1', 'act2', 'act3', 'act4', 'act5', 'bn1', 'bn2', 'bn3', 'bn4', 'bn5', 'bn6', 'conv1', 'conv2', 'conv3', 'conv4', 'dense5', 'dense6', 'flatten_1', 'pool2', 'pool4']
print(weigths_lists)

model = Binary_Net(kernel_size=kernel_size,img_rows=img_rows,img_cols=img_cols, channels=channels,
data_format=data_format,H=H,kernel_lr_multiplier=kernel_lr_multiplier,use_bias=use_bias,
epsilon=epsilon,momentum=momentum,classes=classes,pool_size=pool_size)

# # # serialize model to JSON
# # model_json = model.to_json()
# # with open("model.json", "w") as json_file:
# # 	json_file.write(model_json)
# # json_file = open('model.json', 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()
# # model = model_from_json(loaded_model_json)

# model.load_weights(weights_path)
# config= model.get_config()
# config['layers'][0]['config']['dtype'] = 'float16'
# model = model.from_config(config)
for i in range(len(weigths_lists)):
	print(weigths_lists[i])
	if((i>=11) and (i<=16)):	#只对 【'conv1', 'conv2', 'conv3', 'conv4', 'dense5', 'dense6'】这几个层进行二值化，因为训练时只有这几个层进行了 Binary
		old_weights = model.get_layer(weigths_lists[i]).get_weights()
		new_weights = model.get_layer(weigths_lists[i]).set_weights(SignNumpy(old_weights))
		print(model.get_layer(weigths_lists[i]).get_weights())
model.save_weights('logs/weights/BinaryNet_BinaryWeights_0.99.h5')

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()


# [3]: 进行修改后的权重文件效果评估
start = time.time()
score = model.evaluate(X_test, Y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('%.2f'% (time.time()-start))
###################################### Binary后和原结果分数一致 ######################################