#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
import cv2
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import log_loss
import tensorflow as tf
import datetime
import os
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from keras import backend as K
import tables

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# create a vggface model
vggFace = VGGFace(model='vgg16')

def prepare_dataset():
	images_folder = 'celeba-dataset/img_align_celeba/img_align_celeba'
	df_attr = pd.read_csv('celeba-dataset/list_attr_celeba.csv')
	no_glasses = df_attr['Eyeglasses']<0
	no_hat = df_attr['Wearing_Hat']<0
	young = df_attr['Young']>0
	df_young = df_attr[young & no_hat & no_glasses]
	df_young = df_young[['image_id']]

	# split data into train, val, and test
	input_train, input_test = split_data(df_young['image_id'].values.reshape((df_young.shape[0],1)), 0.6)
	input_val, input_test = split_data(input_test, 0.5)

	# load pics into .npy file
	get_pics_input(images_folder, input_train, 'input_pics_train')
	get_pics_input(images_folder, input_test, 'input_pics_test')
	get_pics_input(images_folder, input_val, 'input_pics_val')

def split_data(input_data, train_percentage):
	indices = list(range(input_data.shape[0]))
	np.random.shuffle(indices)

	num_train = int(input_data.shape[0]*train_percentage)
	train_indices = indices[:num_train]
	test_indices = indices[num_train:]

	return input_data[train_indices,:], input_data[test_indices,:]

def get_pics_input(folder, input_train, filename):
	img_dtype = tables.FloatAtom()
	data_shape = (0, 256, 256, 3)
	hdf5_file = tables.open_file('{}.hdf5'.format(filename), mode='w')
	storage = hdf5_file.create_earray(hdf5_file.root, 'images', img_dtype, shape=data_shape)

	for i in range(input_train.shape[0]):
		print(i)
		filename = '{}/{}'.format(folder, input_train[i][0])
		img = cv2.imread(filename) / 255.0
		img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
		# face = extract_face(img, required_size=(256,256))
		storage.append(img[None])
	hdf5_file.close()

# extract a single face from a given photograph
def extract_face(pic_array, required_size=(224, 224)):
	try:
		# create the detector, using default weights
		detector = MTCNN()
		# detect faces in the image
		results = detector.detect_faces(pic_array)
		# extract the bounding box from the first face
		x1, y1, width, height = results[0]['box']
		x2, y2 = x1 + width, y1 + height
		# extract the face
		face = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = cv2.resize(face, required_size, cv2.INTER_LINEAR)
		print("YAY")
		return image
	except:
		return cv2.resize(pic_array, required_size, cv2.INTER_LINEAR)

def build_transform_model():
	input_layer = tf.keras.layers.Input(shape=(256,256,3), name='inputs')
	conv1 = tf.keras.layers.Conv2D(32, kernel_size=(9,9), strides=1, padding='same', name='conv1')(input_layer)
	batch_norm1 = tf.keras.layers.BatchNormalization()(conv1)
	relu1 = tf.keras.layers.Activation('relu')(batch_norm1)
	conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=2, padding='same', name='conv2')(relu1)
	batch_norm2 = tf.keras.layers.BatchNormalization()(conv2)
	relu2 = tf.keras.layers.Activation('relu')(batch_norm2)
	conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=2, padding='same', name='conv3')(relu2)
	batch_norm3 = tf.keras.layers.BatchNormalization()(conv3)
	relu3 = tf.keras.layers.Activation('relu')(batch_norm3)

	res1_conv1 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', name='rb1')(relu3)
	res1_bn1 = tf.keras.layers.BatchNormalization()(res1_conv1)
	res1_relu = tf.keras.layers.Activation('relu')(res1_bn1)
	res1_conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same')(res1_relu)
	res1_bn2 = tf.keras.layers.BatchNormalization()(res1_conv2)

	res2_conv1 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', name='rb2')(res1_bn2)
	res2_bn1 = tf.keras.layers.BatchNormalization()(res2_conv1)
	res2_relu = tf.keras.layers.Activation('relu')(res2_bn1)
	res2_conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same')(res2_relu)
	res2_bn2 = tf.keras.layers.BatchNormalization()(res2_conv2)

	res3_conv1 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', name='rb3')(res2_bn2)
	res3_bn1 = tf.keras.layers.BatchNormalization()(res3_conv1)
	res3_relu = tf.keras.layers.Activation('relu')(res3_bn1)
	res3_conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same')(res3_relu)
	res3_bn2 = tf.keras.layers.BatchNormalization()(res3_conv2)

	res4_conv1 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', name='rb4')(res3_bn2)
	res4_bn1 = tf.keras.layers.BatchNormalization()(res4_conv1)
	res4_relu = tf.keras.layers.Activation('relu')(res4_bn1)
	res4_conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same')(res4_relu)
	res4_bn2 = tf.keras.layers.BatchNormalization()(res4_conv2)

	res5_conv1 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', name='rb5')(res4_bn2)
	res5_bn1 = tf.keras.layers.BatchNormalization()(res5_conv1)
	res5_relu = tf.keras.layers.Activation('relu')(res5_bn1)
	res5_conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same')(res5_relu)
	res5_bn2 = tf.keras.layers.BatchNormalization()(res5_conv2)

	conv4 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), strides=2, padding='same', name='conv4')(res5_conv2)
	batch_norm4 = tf.keras.layers.BatchNormalization()(conv4)
	relu4 = tf.keras.layers.Activation('relu')(batch_norm4)
	conv5 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='same', name='conv5')(relu4)
	batch_norm5 = tf.keras.layers.BatchNormalization()(conv5)
	relu5 = tf.keras.layers.Activation('relu')(batch_norm5)
	conv6 = tf.keras.layers.Conv2DTranspose(3, kernel_size=(9,9), strides=1, padding='same', name='conv6')(relu5)
	batch_norm6 = tf.keras.layers.BatchNormalization()(conv6)
	relu6 = tf.keras.layers.Activation('relu')(batch_norm6)

	model = tf.keras.Model(inputs=input_layer, outputs=relu6, name='img_transform_model')
	return model

def feature_reconstruction_loss(dim, activation_predicted, activation_target):
	C, H, W = dim
	return 1/(C*H*W) * (np.linalg.norm(activation_predicted - activation_target) ** 2)

def style_reconstruction_loss(dim, activation_predicted, activation_target):
	C, H, W = dim
	reshaped_predicted = np.reshape(activation_predicted, (C, H*W))
	gram_matrix_predicted = 1/(C*H*W) * np.matmul(reshaped_predicted, np.transpose(reshaped_predicted))
	reshaped_target = np.reshape(activation_target, (C, H*W))
	gram_matrix_target = 1/(C*H*W) * np.matmul(reshaped_target, np.transpose(reshaped_target))
	return np.linalg.norm(gram_matrix_predicted - gram_matrix_target) ** 2


def overall_transform_loss_fxn(target_img, transformed_img):
	# convert one face into samples
	# pixels_transformed = transformed_img.astype('float32')
	# samples_transformed = tf.expand_dims(transformed_img, axis=0)
	transformed_img_arr = K.eval(transformed_img)
	target_img_arr = K.eval(target_img)
	print("AHHHHHHHHHHHHHH: {}".format(transformed_img_arr.shape)) #(?, 256, 256, 3)
	print("AHHHHHHHHHHHHHH: {}".format(target_img_arr.shape)) #(?, ?, ?, ?)
	# prepare the face for the model, e.g. center pixels
	samples_transformed = preprocess_input(transformed_img_arr, version=1)

	# perform prediction
	yhat_transformed = vggFace.predict(samples_transformed)

	# get activations for layers from vgg16
	vgg_relu2_2_transformed = vggFace.get_layer('conv2_2').output # for feature reconstruction loss too
	vgg_relu1_2_transformed = vggFace.get_layer('conv1_2').output
	vgg_relu3_3_transformed = vggFace.get_layer('conv3_3').output
	vgg_relu4_3_transformed = vggFace.get_layer('conv4_3').output

	# pixels_target = target_img.astype('float32')
	# samples_target = np.expand_dims(pixels_target, axis=0)
	samples_target = preprocess_input(target_img_arr, version=1)
	yhat_target = vggFace.predict(samples_target)

	vgg_relu2_2_target = vggFace.get_layer('conv2_2').output # for feature reconstruction loss too
	vgg_relu1_2_target = vggFace.get_layer('conv1_2').output
	vgg_relu3_3_target = vggFace.get_layer('conv3_3').output
	vgg_relu4_3_target = vggFace.get_layer('conv4_3').output  

	feature_loss = feature_reconstruction_loss(vgg_relu2_2_target.shape, vgg_relu2_2_transformed, vgg_relu2_2_target)
	style_loss_1 = style_reconstruction_loss(vgg_relu1_2_target.shape, vgg_relu1_2_transformed, vgg_relu1_2_target)
	style_loss_2 = style_reconstruction_loss(vgg_relu2_2_target.shape, vgg_relu2_2_transformed, vgg_relu2_2_target)
	style_loss_3 = style_reconstruction_loss(vgg_relu3_3_target.shape, vgg_relu3_3_transformed, vgg_relu3_3_target)
	style_loss_4 = style_reconstruction_loss(vgg_relu4_3_target.shape, vgg_relu4_3_transformed, vgg_relu4_3_target)

	return feature_loss + style_loss_1 + style_loss_2 + style_loss_3 + style_loss_4

# put data into .npy files for faster loading
# prepare_dataset()

# load prepared data
hdf5_file_train = tables.open_file('input_pics_train.hdf5', mode='r')
input_pics_train = hdf5_file_train.root.images
# print(input_pics_train.shape)
hdf5_file_val = tables.open_file('input_pics_val.hdf5', mode='r')
input_pics_val = hdf5_file_val.root.images
# print(input_pics_val.shape)
hdf5_file_test = tables.open_file('input_pics_test.hdf5', mode='r')
input_pics_test = hdf5_file_test.root.images
# print(input_pics_test.shape)

# just for testing on a small scale
input_pics_train = input_pics_train[:5,:]
input_pics_val = input_pics_val[:5,:]
input_pics_test = input_pics_test[:5,:]


# build image transform network
target_einstein = cv2.resize(cv2.imread('target_einstein.jpg'), (256,256), cv2.INTER_LINEAR)
transform_model = build_transform_model()

expanded_einstein = np.expand_dims(target_einstein, axis=0)
tiled_einstein_train = tf.tile(expanded_einstein, (input_pics_train.shape[0], 1, 1, 1))
tiled_einstein_val = tf.tile(expanded_einstein, (input_pics_val.shape[0], 1, 1, 1))

for layer in transform_model.layers:
	print('{}: {}'.format(layer.name, layer.output.shape))

lr = 1e-2
transform_model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mean_squared_error')#overall_transform_loss_fxn)
logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)
checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'transform_network_weights.h5'),
                                                        monitor='val_loss',
                                                        verbose=0,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        period=1)
transform_model.fit(input_pics_train, tiled_einstein_train, epochs=2, batch_size=4, steps_per_epoch=8, validation_steps=32,
             validation_data=(input_pics_val, tiled_einstein_val),
             callbacks=[tbCallBack, checkpointCallBack])
predicted_targets = transform_model.predict(np.expand_dims(input_pics_test[0], axis=0), batch_size=4, steps=64)
cv2.imshow('',input_pics_test[0])
cv2.waitKey(0)
print(predicted_targets.shape)
cv2.imshow('',predicted_targets[0])
cv2.waitKey(0)