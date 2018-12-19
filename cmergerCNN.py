from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time

import tensorflow as tf

import os

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from tensorflow.python.client import device_lib

import keras
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

from keras.legacy.layers import MaxoutDense

import math

from keras import callbacks as cb

from keras.utils import plot_model

X_train = np.load('procdata/merger_crop_4bands_c.npy')
Y_train = np.load('procdata/merger_crop_4bands_c_labels.npy')

classWeight = compute_class_weight('balanced', np.array([0, 1]), Y_train) 
classWeight = dict(enumerate(classWeight))

idjx = np.random.choice(2, size=len(X_train)).astype(bool)

idx = np.zeros_like(Y_train)
idx[:len(idjx)] = idjx
idx=idx.astype(bool)

x_test = X_train[idx]
y_test = Y_train[idx]
x_train = X_train[~idx]
y_train = Y_train[~idx]

from sklearn.model_selection import train_test_split
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)



def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 100.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate


lrate = cb.LearningRateScheduler(step_decay)

def vgglike():
	cnnmodel = Sequential()

	regularizer = None#regularizers.l2(0.1)
	activity = None#regularizers.l1(0.1)

	cnnmodel.add(Conv2D(64, (3, 3), padding='same', input_shape=(64, 64, 4), activity_regularizer=activity, kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(MaxPooling2D(2,2))
	
	cnnmodel.add(Conv2D(128, (3, 3), padding='same',  kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(MaxPooling2D(2, 2))

	cnnmodel.add(Conv2D(256, (3, 3), padding='same',  kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())		
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(MaxPooling2D(2, 2))
	
	cnnmodel.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(MaxPooling2D(2, 2))
	cnnmodel.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer))
	cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))

	cnnmodel.add(Flatten())	
	cnnmodel.add(Dense(2048))
	cnnmodel.add(Dropout(0.5))
	#cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(MaxoutDense(2048))
	cnnmodel.add(Dropout(0.5))
	#cnnmodel.add(BatchNormalization())
	cnnmodel.add(Activation('relu'))
	cnnmodel.add(MaxoutDense(2))
	cnnmodel.add(Activation('sigmoid'))
	
	cnnmodel.compile(loss='binary_crossentropy', optimizer=SGD(momentum=0.9, nesterov=True),
                metrics=['accuracy'])
	cnnmodel.summary()
	cnnmodel.save_weights('cnnmodel_init_weights.tf')
	return cnnmodel

cnnmodel = vgglike()


plot_model(cnnmodel, to_file='model.png')


batch_size = 128
epochs = 250 

tensorboard = TensorBoard(log_dir='logs/{}.log'.format(time.time()))


datagen = ImageDataGenerator(
    rotation_range=360, width_shift_range=4, height_shift_range=4, fill_mode='nearest')


history = cnnmodel.fit_generator(datagen.flow(x_train, y_train,
                       batch_size=batch_size), 
                      epochs=4*epochs,
                      verbose=2,
                      validation_data=(x_test, y_test), class_weight=classWeight)





cnnmodel.save('4vgg4bandsmodel.h5')
val_predictions = cnnmodel.predict(x_val)
np.save('val_pred.npy', val_predictions)
np.save('xvals.npy', x_val)
np.save('yvals.npy', y_val)
all_predicts = cnnmodel.predict(X_train)
np.save('pred.npy', all_predicts)



def histplot(history):
    hist = pd.DataFrame(history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    hist.plot(y=['loss', 'val_loss'], ax=ax1)
    min_loss = hist['val_loss'].min()
    ax1.hlines(min_loss, 0, len(hist), linestyle='dotted',
               label='min(val_loss) = {:.3f}'.format(min_loss))
    ax1.legend(loc='upper right')
    hist.plot(y=['acc', 'val_acc'], ax=ax2)
    max_acc = hist['val_acc'].max()
    ax2.hlines(max_acc, 0, len(hist), linestyle='dotted',
               label='max(val_acc) = {:.3f}'.format(max_acc))
    ax2.legend(loc='lower right', fontsize='large')
    fig.savefig('hist_{}.png'.format(time.time()))

histplot(history)

