import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50     import ResNet50
from tensorflow.keras.applications.resnet50     import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.datasets import cifar10


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import numpy as np 
import os 
import sys 
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def step_decay(epoch):
	initial_lrate = 1e-3
	drop = 0.75
	epochs_drop = 3.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


def triplet_loss(y_true, y_pred, alpha = 0.2):
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
    
    total_lenght = y_pred.shape.as_list()[-1]
#     print('total_lenght=',  total_lenght)
#     total_lenght =12
    
    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss


## 
## https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
def lossless_triplet_loss(y_true, y_pred, epsilon=1e-8):
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
    beta = total_length / 3
    anchor = y_pred[:,0:int(total_length*1/3)]
    positive = y_pred[:,int(total_length*1/3):int(total_length*2/3)]
    negative = y_pred[:,int(total_length*2/3):int(total_length*3/3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # -ln(-x/N+1)
    pos_dist = -tf.math.log(-tf.divide((pos_dist), beta)+1+epsilon)
    neg_dist = -tf.math.log(-tf.divide((beta - neg_dist), beta)+1+epsilon)
    
    # compute loss
    loss = neg_dist + pos_dist
    
    return loss

def create_base_network(img_d = 224, img_c = 3):
    """
    Base network to be shared.
    ## input_dim = (224, 224. 3)
    """
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_d, img_d, img_c))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='softmax'))
    print(model.summary())
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
    #model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss=triplet_loss)
    model.compile(optimizer=optimizers.RMSprop(lr=1e-3), loss=triplet_loss)
    return model


def lr_schedule(epoch):
    if epoch <= 5:
        lrate = 1e-3  
    elif epoch > 5 and epoch <= 12 :
        lrate = 1e-4  
    elif epoch > 12 and epoch <= 25 :
        lrate = 1e-5   
    elif epoch > 25:
        lrate =1e-5
    return lrate


def embedding_analysis(reference_embeddings, test_embeddings, reference_categories):
    test_embeddings = np.vstack(test_embeddings)
    reference_embeddings = np.vstack(reference_embeddings)              
    dist = pairwise_distances(reference_embeddings, test_embeddings)
    for i in range(len(reference_embeddings)):
        print("Categories : ", reference_categories[i])
    print("Distance : ")
    print(dist)
    print("closest to: ")
    for i in range(len(reference_embeddings)):
        print(np.argmin(dist[i]))
    for i in range(len(reference_embeddings)):
        print("Test Category (Predicted)", reference_categories[np.argmin(dist[i])],
              ", Test Category : ", reference_categories[i])

def images_collage(reference_images, test_images):
    fig, ax = plt.subplots(2, len(reference_images), figsize = (3 * len(reference_images), 6))
    for i, axi in enumerate(ax.flat):
        if i < len(reference_images):
            axi.imshow(x_train[reference_images[i]])      
        else:
            axi.imshow(x_test[test_images[i - len(reference_images)]])
    plt.savefig("TrainTestImages")

def images_collage_only_test(reference_images, test_images):
    fig, ax = plt.subplots(2, len(reference_images), figsize = (3 * len(reference_images), 4))
    for i, axi in enumerate(ax.flat):
        if i < len(reference_images):
            axi.imshow(x_test[reference_images[i]])      
        else:
            axi.imshow(x_test[test_images[i - len(reference_images)]])
    plt.savefig("TestTestImages-64" + str(len(reference_images)))

def use_case_one(encoder_network):
    ## networks ability to find similiarty between never seen test images with training images 
    ## plane, car, ship truck
    reference_images = [77, 46, 92, 1]
    reference_categories = ['Airplane', 'Cars', 'Ship', 'Truck']

    reference_embeddings = []
    for idx in reference_images:
        train_img = x_train[idx].reshape(-1, 32,32, 3)
        upscaled_ref_img =  tf.keras.backend.resize_images(train_img, 
                                                    height_factor = 7, 
                                                    width_factor = 7, 
                                                    data_format = "channels_last")
        pred = encoder_network.predict(upscaled_ref_img)
        reference_embeddings.append(pred)
    ## Let's sample 4 images from test datasets - These images has never been seen by the network before
    rand_idx = random.randint(1, 1001)
    print(rand_idx)
    plane_index = np.where(y_test[:,0] == 0)[0][rand_idx]
    cars_index = np.where(y_test[:,0] == 1)[0][rand_idx]
    ship_index = np.where(y_test[:,0] == 8)[0][rand_idx]
    truck_index = np.where(y_test[:,0] == 9)[0][rand_idx]
    test_images = [plane_index, cars_index, ship_index, truck_index]
    print("Test Images index :", test_images)
    test_categories = ['Airplane', 'Cars', 'Ship', 'Truck']
    test_embeddings = []
    for idx in test_images:
        test_img = x_test[idx].reshape(-1, 32,32, 3)
        test_upscaled_img =  tf.keras.backend.resize_images(test_img, 
                                                    height_factor = 7, 
                                                    width_factor = 7, 
                                                    data_format = "channels_last")
        pred = encoder_network.predict(test_upscaled_img)
        test_embeddings.append(pred)
    images_collage(reference_images, test_images)
    embedding_analysis(reference_embeddings, test_embeddings, reference_categories)

def use_case_two(encoder_network):
    ## networks ability to find similiarty between never seen test images with training images 
    ## plane, car, ship truck

    bird_index = np.where(y_test[:,0] == 2)
    cat_index = np.where(y_test[:,0] == 3)
    deer_index = np.where(y_test[:,0] == 4)
    dog_index = np.where(y_test[:,0] == 5)
    frog_index = np.where(y_test[:,0] == 6)
    horse_index = np.where(y_test[:,0] == 7)
    ref_random = random.randint(1, 1001)
    ref_img_idx = [bird_index[0][ref_random],
                   cat_index[0][ref_random],
                   deer_index[0][ref_random],
                   dog_index[0][ref_random],
                   frog_index[0][ref_random],
                   horse_index[0][ref_random] ]

    bird_ref_img = x_test[bird_index[0][ref_random]]
    cat_ref_img = x_test[cat_index[0][ref_random]]
    deer_ref_img = x_test[deer_index[0][ref_random]]
    dog_ref_img = x_test[dog_index[0][ref_random]]
    frog_ref_img = x_test[frog_index[0][ref_random]]
    horse_ref_img = x_test[horse_index[0][ref_random]]

    print(bird_ref_img.shape)
    reference_categories = ['Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse']
    ref_imgs = [bird_ref_img, cat_ref_img, deer_ref_img, dog_ref_img, frog_ref_img, horse_ref_img]

    reference_embeddings = []
    for img in ref_imgs:
        train_img = img.reshape(-1, 32,32, 3)
        upscaled_ref_img =  tf.keras.backend.resize_images(train_img, 
                                                    height_factor = 7, 
                                                    width_factor = 7, 
                                                    data_format = "channels_last")
        pred = encoder_network.predict(upscaled_ref_img)
        reference_embeddings.append(pred)

    reference_embeddings = np.vstack(reference_embeddings)
    print(reference_embeddings.shape)

    test_embeddings = []
    ## randomly pick an image for each label
    rand_idx = random.randint(1, 1001)
    while(rand_idx == ref_random):
        rand_idx = random.randint(1, 1001)
    bird_test_img = x_test[bird_index[0][rand_idx]]
    cat_test_img = x_test[cat_index[0][rand_idx]]
    deer_test_img = x_test[deer_index[0][rand_idx]]
    dog_test_img = x_test[dog_index[0][rand_idx]]
    frog_test_img = x_test[frog_index[0][rand_idx]]
    horse_test_img = x_test[horse_index[0][rand_idx]]

    test_img_idx = [bird_index[0][rand_idx],
                   cat_index[0][rand_idx],
                   deer_index[0][rand_idx],
                   dog_index[0][rand_idx],
                   frog_index[0][rand_idx],
                   horse_index[0][rand_idx] ]
    print(bird_test_img.shape)
    test_imgs = [bird_test_img, cat_test_img, deer_test_img, dog_test_img, frog_test_img, horse_test_img]
    test_embeddings = []
    for img in test_imgs:
        test_img = img.reshape(-1, 32,32, 3)
        upscaled_test_img =  tf.keras.backend.resize_images(test_img, 
                                                    height_factor = 7, 
                                                    width_factor = 7, 
                                                    data_format = "channels_last")
        pred = encoder_network.predict(upscaled_test_img)
        test_embeddings.append(pred)

    test_embeddings = np.vstack(test_embeddings)
    print(test_embeddings.shape)
    images_collage_only_test(ref_img_idx, test_img_idx)
    embedding_analysis(reference_embeddings, test_embeddings, reference_categories)

################# main program ########################

TRAIN = 1
VAL = 2


if __name__ == "__main__":
    if sys.argv[1].lower() == "train":
        print("Starting Training... ")
        op = TRAIN
    else:
        print("Starting Eval...")
        op = VAL
    if op == TRAIN:
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

        cp = ModelCheckpoint('cifar_tsne_wts-64.h5',
                                monitor='loss',
                                save_best_only=True,
                                verbose=1,
                                mode='auto',
                                save_weights_only=True)

        lr = LearningRateScheduler(lr_schedule)
        #lr = LearningRateScheduler(step_decay)

        hist = tsnn_model.fit([Anchor,Positive,Negative],y=Y_dummy,
                validation_data=([Anchor_test,Positive_test,Negative_test],Y_dummy2), 
                shuffle = True,
                batch_size=20, epochs=25,
                callbacks = [cp, lr])
        
        tsnn_model.save('cifar_tsne_model_v2-64.h5')
    if op == VAL:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        cifar_classes = [ 'Airplane',
                    'Automobile',
                    'Bird',
                    'Cat',
                    'Deer',
                    'Dog',
                    'Frog',
                    'Horse',
                    'Ship',
                    'Truck']

        
        encoder_network = create_base_network()
        encoder_network.load_weights('cifar_tsne_wts-64.h5', by_name = 'sequential')
        print(encoder_network.summary())

        #use_case_one(encoder_network)
        use_case_two(encoder_network)