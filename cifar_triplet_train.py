import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50     import ResNet50
from tensorflow.keras.applications.resnet50     import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D, concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np 

def triplet_loss(y_true, y_pred, alpha = 0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    
    total_length = y_pred.shape.as_list()[-1]
#     print('total_lenght=',  total_lenght)
#     total_lenght =12
    
    anchor = y_pred[:,0:int(total_lentht*1/3)]
    positive = y_pred[:,int(total_length*1/3):int(total_length*2/3)]
    negative = y_pred[:,int(total_length*2/3):int(total_length*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss

def create_base_network():
    """
    Base network to be shared.
    ## input_dim = (224, 224. 3)
    """
    model = load_model('cifar_resnet_model_v2.h5')    
    return model


def tsnn_network(img_d = 224, img_c = 3):
    anchor_input = Input((img_d, img_d, img_c, ), name='anchor_input')
    positive_input = Input((img_d, img_d, img_c, ), name='positive_input')
    negative_input = Input((img_d, img_d, img_c, ), name='negative_input')

    # Shared embedding layer for positive and negative items
    Shared_DNN = create_base_network()
    encoded_anchor = Shared_DNN(anchor_input)
    encoded_positive = Shared_DNN(positive_input)
    encoded_negative = Shared_DNN(negative_input)

    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

    model = Model(inputs=[anchor_input,positive_input, negative_input], outputs=merged_vector)
    adam_optim = Adam(lr=0.0001, decay = 1e-6)
    model.compile(loss=triplet_loss, optimizer=adam_optim)
    return model

if __name__ == "__main__":
    img_lw = 32
    img_ch = 3
    ## load the predefined triplets from npy files
    print("Loading triplets..")
    tsnn_train_triplets = np.load('tsnn_train_triplets.npy')
    tsnn_test_triplets = np.load('tsnn_test_triplets.npy')
    print(tsnn_train_triplets.shape)
    print(tsnn_test_triplets.shape)

    Anchor = tsnn_train_triplets[:,0,:].reshape(-1,img_lw,img_lw,img_ch,)
    Positive = tsnn_train_triplets[:,1,:].reshape(-1,img_lw,img_lw,img_ch,)
    Negative = tsnn_train_triplets[:,2,:].reshape(-1,img_lw,img_lw,img_ch,)
    Anchor_test = tsnn_test_triplets[:,0,:].reshape(-1,img_lw,img_lw,img_ch,)
    Positive_test = tsnn_test_triplets[:,1,:].reshape(-1,img_lw,img_lw,img_ch,)
    Negative_test = tsnn_test_triplets[:,2,:].reshape(-1,img_lw,img_lw,img_ch,)

    print("Upscaling images.")
    Anchor = tf.keras.backend.resize_images(Anchor, 
                                                  height_factor = 7, 
                                                  width_factor = 7, 
                                                  data_format = "channels_last")
    Positive = tf.keras.backend.resize_images(Positive, 
                                                  height_factor = 7, 
                                                  width_factor = 7, 
                                                  data_format = "channels_last")
    Negative = tf.keras.backend.resize_images(Negative, 
                                                  height_factor = 7, 
                                                  width_factor = 7, 
                                                  data_format = "channels_last")

    Anchor_test = tf.keras.backend.resize_images(Anchor_test, 
                                                  height_factor = 7, 
                                                  width_factor = 7, 
                                                  data_format = "channels_last")
    Positive_test = tf.keras.backend.resize_images(Positive_test, 
                                                  height_factor = 7, 
                                                  width_factor = 7, 
                                                  data_format = "channels_last")
    Negative_test = tf.keras.backend.resize_images(Negative_test, 
                                                  height_factor = 7, 
                                                  width_factor = 7, 
                                                  data_format = "channels_last")
    print("Applying ResNet50 preprocessing")
    ## apply resnet50 preprocessing
    Anchor = resnet50_preprocess_input(Anchor)
    Positive = resnet50_preprocess_input(Positive)
    Negative = resnet50_preprocess_input(Negative)
    Anchor_test = resnet50_preprocess_input(Anchor_test)
    Positive_test = resnet50_preprocess_input(Positive_test)
    Negative_test = resnet50_preprocess_input(Negative_test)
    
    print(Anchor.shape, Positive.shape, Negative.shape)
    print(Anchor_test.shape, Positive_test.shape, Negative_test.shape)
    Y_dummy = np.empty((Anchor.shape[0],300))
    Y_dummy2 = np.empty((Anchor_test.shape[0],1))
    tsnn_model = tsnn_network()
    print(tsnn_model.summary())

    cp = ModelCheckpoint('cifar_tsne_wts.h5',
                            monitor='loss',
                            save_best_only=True,
                            verbose=1,
                            mode='auto',
                            save_weights_only=True)


    hist = tsnn_model.fit([Anchor,Positive,Negative],y=Y_dummy,
            validation_data=([Anchor_test,Positive_test,Negative_test],Y_dummy2), 
            shuffle = True,
            batch_size=16, epochs=10,
            callbacks = [cp])

    tsnn_model.save('cifar_tsne_model_v2.h5')