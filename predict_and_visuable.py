from __future__ import print_function

import numpy as np 
import cv2
import sys  
sys.path.append('./models')
import keras
from keras.models import Model
from feature_visualize import get_row_col,visualize_feature_map

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
from model import Binary_Net
from config import *


def rotate(image, angle, center=None, scale=1.0):
	(h, w) = image.shape[:2] 
	if center is None: 
		center = (w // 2, h // 2) 
	M = cv2.getRotationMatrix2D(center, angle, scale) 
	rotated = cv2.warpAffine(image, M, (w, h)) 
	return rotated 


def predict(image_path,TTA=True):
	# build a model
	model = Binary_Net(kernel_size=kernel_size,img_rows=img_rows,img_cols=img_cols, channels=channels,
					data_format=data_format,H=H,kernel_lr_multiplier=kernel_lr_multiplier,use_bias=use_bias,
					epsilon=epsilon,momentum=momentum,classes=classes,pool_size=pool_size)

	model.load_weights('logs/weights/Binary_Net_20_0.99.h5')
	#model = Model(inputs=model.input, outputs=model.get_layer('residual_attention_stage1').output)
	labels = ['0','1','2','3','4','5','6','7','8','9']

	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

	if(TTA):
		h_flip = cv2.flip(image, 1)		# 水平翻转
		v_flip = cv2.flip(image, 0)		# 垂直翻转
		rotated45 = rotate(image, 45)	#旋转
		rotated90 = rotate(image, 90)
		rotated180 = rotate(image, 180)
		rotated270 = rotate(image, 270)
		image_list = []
		image_list.append(h_flip)
		image_list.append(v_flip)
		image_list.append(rotated45)
		image_list.append(rotated90)
		image_list.append(rotated180)
		image_list.append(rotated270)

		pred_list = []
		for i in range(len(image_list)):
			test = cv2.resize(image_list[i], (28, 28))	#将测试图片转化为model需要的大小
			test = np.array(test, np.float32) / 255	#归一化送入，float/255
			test = test.reshape(1,28,28,1)	#model需要的是1张input_size*input_size得3通道rgb图片，所以转化为(1,input_size,input_size,3)

			pred = model.predict(test)
			pred_list.append(pred)

		TTA_pred = np.zeros(shape=(1,10))
		for i in range(len(pred_list)):
			TTA_pred = TTA_pred+pred_list[i]

		print('TTA_pred:',TTA_pred,'pred_shape:',TTA_pred.shape)
		max_score = np.where(TTA_pred==np.max(TTA_pred))
		label = labels[int(max_score[1])]
		print(label)

	else:
		test = cv2.resize(image, (28, 28))	#将测试图片转化为model需要的大小
		test = np.array(test, np.float32) / 255	#归一化送入，float/255
		test = test.reshape(1,28,28,1)	#model需要的是1张input_size*input_size得3通道rgb图片，所以转化为(1,input_size,input_size,3)

		pred = model.predict(test)
		print('prediction:',pred,'pred_shape:',pred.shape)

		max_score = np.where(pred==np.max(pred))
		print(max_score)
		label = labels[int(max_score[1])]
		print(label)


def visuable(image_path,name):
	# build a model
	model = Binary_Net(kernel_size=kernel_size,img_rows=img_rows,img_cols=img_cols, channels=channels,
					data_format=data_format,H=H,kernel_lr_multiplier=kernel_lr_multiplier,use_bias=use_bias,
					epsilon=epsilon,momentum=momentum,classes=classes,pool_size=pool_size)
	model.load_weights('logs/weights/Binary_Net_20_0.99.h5')
	model = Model(inputs=model.input, outputs=model.get_layer('conv1').output)
	labels = ['0','1','2','3','4','5','6','7','8','9']

	test = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	test = cv2.resize(test, (28, 28))	#将测试图片转化为model需要的大小
	test = np.array(test, np.float32) / 255	#归一化送入，float/255
	test = test.reshape(1,28,28,1)	#model需要的是1张input_size*input_size得1通道rgb图片，所以转化为(1,input_size,input_size,3)
	block_pool_features = model.predict(test)
	print(block_pool_features.shape)

	feature = block_pool_features.reshape(block_pool_features.shape[1:])
	visualize_feature_map(feature,name)

if __name__ == '__main__':
	import tensorflow as tf
	from keras.backend.tensorflow_backend import set_session
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	set_session(tf.Session(config=config))

	import os
	import random
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	file_names = next(os.walk('test_images'))[2]
	file_names = random.choice(file_names)
	filepath = os.path.join('test_images',file_names)
	print(filepath)
	predict(image_path=filepath,TTA=False)
	visuable(image_path=filepath,name=file_names.split('.')[0])
	# model = Binary_Net(kernel_size=kernel_size,img_rows=img_rows,img_cols=img_cols, channels=channels,
	# 				data_format=data_format,H=H,kernel_lr_multiplier=kernel_lr_multiplier,use_bias=use_bias,
	# 				epsilon=epsilon,momentum=momentum,classes=classes,pool_size=pool_size)
	# model.load_weights('logs/weights/Binary_Net_20_0.99.h5')
	# model = Model(inputs=model.input, outputs=model.get_layer('conv1').output)
	# labels = ['0','1','2','3','4','5','6','7','8','9']
	# weight_Dense_1 = model.get_layer('conv1').get_weights()
	# print(weight_Dense_1)
	# print(weight_Dense_1.shape)
	# print(bias_Dense_1.shape)


