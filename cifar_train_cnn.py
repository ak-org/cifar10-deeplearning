'''
Helpful link https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint


 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 10 and epoch <=25:
        lrate = 1e-4
    elif epoch > 25:
        lrate = 1e-5
    return lrate

def hand_crafted_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    print(model.summary())
    return model	

es = EarlyStopping(monitor='val_loss', 
                    mode='min', 
                    verbose=1, 
                    patience=50)

cp = ModelCheckpoint('cifar_resnet_model_v1.h5', 
                     monitor='val_loss', verbose=1, 
                     save_best_only=True, 
                     save_weights_only = False, 
                     mode='min')

opt_rms = tf.keras.optimizers.RMSprop(lr=0.001,decay=1e-6)

#data augmentation
datagen = ImageDataGenerator(
            width_shift_range=0.05, #0.1, 0.2
            height_shift_range=0.05, #0.1, 0.2
            #vertical_flip = True,
            )

if __name__ == "__main__":
    #training
    batch_size =50
    TRAIN_SIZE = 40000
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    #z-score
    mean_train = np.mean(x_train,axis=(0,1,2,3))
    std_train = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean_train)/(std_train+1e-7)

    mean_test = np.mean(x_test,axis=(0,1,2,3))
    std_test = np.std(x_test,axis=(0,1,2,3))
    x_test = (x_test-mean_test)/(std_test+1e-7)
    
    num_classes = 10
    y_train = to_categorical(y_train,num_classes)
    y_test = to_categorical(y_test,num_classes)
    
    weight_decay = 0.0001

    datagen.fit(x_train)
    
    crafted_model = hand_crafted_model()
    steps_per_epoch = TRAIN_SIZE // batch_size
    crafted_model.fit_generator(datagen.flow(x_train[0:TRAIN_SIZE], y_train[:TRAIN_SIZE], batch_size=batch_size),
                        steps_per_epoch= steps_per_epoch,
                        epochs=20,
                        verbose=1,
                        shuffle = True,
                        validation_data=(x_train[TRAIN_SIZE:],y_train[TRAIN_SIZE:]),
                        callbacks=[LearningRateScheduler(lr_schedule), cp, es])
'''
last run results 
Epoch 00018: val_loss improved from 0.58126 to 0.57487, saving model to cifar_resnet_model_v1.h5
800/800 [==============================] - 75s 93ms/step - loss: 0.4997 - accuracy: 0.8703 - val_loss: 0.5749 - val_accuracy: 0.8536
Epoch 19/20
799/800 [============================>.] - ETA: 0s - loss: 0.4936 - accuracy: 0.8735  
Epoch 00019: val_loss did not improve from 0.57487
800/800 [==============================] - 74s 93ms/step - loss: 0.4934 - accuracy: 0.8735 - val_loss: 0.5798 - val_accuracy: 0.8504
Epoch 20/20
799/800 [============================>.] - ETA: 0s - loss: 0.4820 - accuracy: 0.8759  
Epoch 00020: val_loss did not improve from 0.57487
800/800 [==============================] - 73s 91ms/step - loss: 0.4820 - accuracy: 0.8760 - val_loss: 0.5759 - val_accuracy: 0.8513

'''