import tensorflow as tf
import numpy as np
import random
import sys 
import os 

from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

num_classes = 10
BATCH_SIZE = 40
TRAIN_SAMPLE = 50000
TRAIN_SIZE = int(TRAIN_SAMPLE * 0.8)
TEST_SIZE = 50000 - TRAIN_SIZE

def list_version():
    print(tf.__version__)
    print(keras.__version__)

def custom_model(img_d = 128, img_c = 3):
    resnet_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (img_d, img_d, img_c))
    model = Sequential()
    model.add(resnet_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer = RMSprop(lr=1e-3), #Adam(lr=1e-4),
              metrics=['accuracy'])
    print(model.summary())
    return model

def cifar_batch_generator(X, y, dim, batch_size = BATCH_SIZE ):
    # Create empty arrays to contain batch of features and labels#
    org_d = 32
    org_c = 3

    while True:
        ## randomly select samples of size BATCH_SIZE 
        idx = np.random.randint(0, dim, size = (batch_size))
        upscaled_X_train = tf.keras.backend.resize_images(X_train[idx].reshape(-1, org_d, org_d, org_c), 
                                                    height_factor = 4, #32 * 7 = 224
                                                    width_factor = 4, 
                                                    data_format = "channels_last")
        upscaled_X_train = preprocess_input(upscaled_X_train)                                            
        yield upscaled_X_train, y[idx]


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print("There are {} train images and {} test images.".format(X_train.shape[0], X_test.shape[0]))
    print('There are {} unique classes to predict.'.format(np.unique(y_train).shape[0]))
    print("Train Size : {}, Test Size : {}".format(TRAIN_SIZE, TEST_SIZE))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print(y_test.shape)
    cifar_model = custom_model()
    checkpoint = ModelCheckpoint('cifar_resnet.h5', monitor='val_loss', verbose=1, 
                                  save_best_only=True, save_weights_only = False, mode='min')
    hist = cifar_model.fit_generator(cifar_batch_generator(X_train[:TRAIN_SIZE], y_train[:TRAIN_SIZE], dim = TRAIN_SIZE), 
                               steps_per_epoch = TRAIN_SIZE // BATCH_SIZE,
                               validation_data = cifar_batch_generator(X_train[TRAIN_SIZE:], y_train[TRAIN_SIZE:], dim = TEST_SIZE),
                               validation_steps = TEST_SIZE // BATCH_SIZE,
                               epochs = 5, verbose = 1,
                               callbacks = [checkpoint])
