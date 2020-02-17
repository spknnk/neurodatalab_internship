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

from train import pad_mel_matrix, get_dataframe

# CONSTANTS
MEL_HEIGHT = 80
MAX_PADDING = 500
EPOCHS = 100  # Early stopping
BATCH = 64

# SEED EVERYTHING
seed_value = 1221
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.set_random_seed(seed_value)

print("Cooking dataset...")
X_test, test_labels = get_dataframe("test")
print("Evaluating...")
model = load_model("model.h5")
scores = model.evaluate(X_test, test_labels)
print("Accuracy: {}".format(scores[1]))
