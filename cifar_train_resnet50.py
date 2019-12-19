import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50     import ResNet50
from tensorflow.keras.applications.vgg16        import VGG16
from tensorflow.keras.applications.vgg19        import VGG19
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_v3 import preprocess_input as incv3_preprocess_input
from tensorflow.keras.applications.resnet50     import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.vgg16        import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19        import preprocess_input as vgg19_preprocess_input

import numpy as np


print(tf.__version__)

def custom_resnet_cifar(img_d = 128, img_c= 3):
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_d, img_d, img_c))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model 

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

onehot_y_train = to_categorical(y_train, 10)
onehot_y_test = to_categorical(y_test, 10)

print( X_train.shape, y_train.shape,  X_test.shape, y_test.shape )

upscaled_x_train = tf.keras.backend.resize_images(X_train, 
                                                  height_factor = 4, 
                                                  width_factor = 4, 
                                                  data_format = "channels_last")

## preprocess input
print("Pre processing inputs")
upscaled_x_train = resnet50_preprocess_input(upscaled_x_train)
print(upscaled_x_train.shape)
print("Building custom model")
custom_model = custom_resnet_cifar()
print("Training")
history = custom_model.fit(upscaled_x_train, onehot_y_train, 
                           epochs=5, batch_size=20,
                           shuffle = True, 
                           validation_split=0.2)
custom_model.save('cifar_resnet_model_v2.h5')

'''
ResNet50
========
Results with image size = 64,64,3 
Epoch 5/5
115s 3ms/sample - loss: 0.1494 - acc: 0.9488 - val_loss: 0.0851 - val_acc: 0.9711

Results with image size = 128,128,3 
Epoch 5/5
Epoch 5/5
204s 5ms/sample - loss: 0.0879 - acc: 0.9729 - val_loss: 0.0425 - val_acc: 0.9866
'''