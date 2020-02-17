# Warning suppression
import warnings
warnings.simplefilter('ignore')
import numpy as np
np.warnings.filterwarnings('ignore')

import math
import random

# Cliche
import os
#import tensorflow as tf
import sys

import keras
import tensorflow as tf
from tqdm import tqdm

from keras.callbacks import LearningRateScheduler
from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
	concatenate, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from keras.models import Model, load_model

from sklearn.metrics import accuracy_score

# CONSTANTS

MEL_HEIGHT = 80
MAX_PADDING = 500
EPOCHS = 100  # Early stopping
BATCH = 64

# Image saved in numpy array have constant height and can have different length due to  
# length of original audio. MAX_PADDING is randomly cut part of audio if its length 
# exceeds this consant and repeat this audio several times if it doesnt

# SEED EVERYTHING

seed_value = 1221
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

### DATA

def pad_mel_matrix(arr, pad_len=MAX_PADDING):
	l = len(arr)
	if l == pad_len:
		return arr
	if l > pad_len:
		idx = np.random.randint(0, l-pad_len)
		return arr[idx:idx+pad_len, :]
	else:
		arr = np.concatenate((arr, arr), axis=0)
		return pad_mel_matrix(arr, pad_len)

def get_dataframe(name): # clean audio is labeled by 0
	"""
	Return dataframe (train/val/test) and corresponding labels
	"""
	name = "./" + name + "/"
	dataframe, tmp = np.array([]), np.array([])
	iterator = 0
	clean_path = os.path.join(name, "clean/")
	noisy_path = os.path.join(name, "noisy/")
	persons_talking = os.listdir(clean_path) # same as noisy
	person_clean__path, person_noisy_path = "", ""

	for customer in tqdm(persons_talking):
		customer_data = np.array([])
		person_clean__path = str(os.path.join(clean_path, customer))
		person_noisy_path = str(os.path.join(noisy_path, customer))
		audiofiles = os.listdir(person_clean__path)
		
		for audio in audiofiles:
			clean_rec = np.load(person_clean__path + "/" + audio)
			noisy_rec = np.load(person_noisy_path + "/" + audio)
			clean_rec, noisy_rec = pad_mel_matrix(clean_rec), pad_mel_matrix(noisy_rec)
			customer_data = np.append(customer_data, clean_rec)
			customer_data = np.append(customer_data, noisy_rec)
			
		tmp = np.append(tmp, customer_data)
		if iterator % 100 == 0 and iterator != 0: # buffer for rare appends
			dataframe = np.append(dataframe, tmp)
			tmp = np.array([])
		iterator += 1
	if tmp.size != 0:
		dataframe = np.append(dataframe, tmp)
	dataframe = dataframe.reshape(-1, MAX_PADDING, MEL_HEIGHT, 1)
	return dataframe, [0, 1] * int(dataframe.shape[0]/2) 

### MODEL

def get_model_mel():

	inp = Input(shape=(MAX_PADDING, MEL_HEIGHT, 1))
	norm_inp = BatchNormalization()(inp)
	img_1 = Convolution2D(32, kernel_size=(25, 4), activation=activations.relu)(norm_inp)
	img_1 = MaxPooling2D(pool_size=(8, 2))(img_1)
	img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
	img_1 = MaxPooling2D(pool_size=(3, 3))(img_1)
	img_1 = GlobalMaxPool2D()(img_1)
	dense_1 = Dense(1, activation=activations.sigmoid)(img_1)

	model = models.Model(inputs=inp, outputs=dense_1)
	opt = optimizers.Adam()

	model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
	model.summary()
	return model

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop,  
		   math.floor((1+epoch)/epochs_drop))
	return lrate

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.lr = []
 
	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.lr.append(step_decay(len(self.losses)))

if __name__ == "__main__":

	print("Cooking datasets...")
	X_train, train_labels = get_dataframe("train")
	X_val, val_labels = get_dataframe("val")
	print("Success!")

	#loss_history = LossHistory()
	#lrate = LearningRateScheduler(step_decay)
	es = EarlyStopping(patience=3, monitor='val_loss', mode='min')
	mcp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
	callbacks_list = [mcp_save, es]

	model = get_model_mel()

	print("Fitting model...")
	model.fit(X_train, train_labels, shuffle=False, epochs=EPOCHS, batch_size=BATCH, verbose=1, validation_data=(X_val, val_labels), callbacks=callbacks_list)
	print("Done!")
