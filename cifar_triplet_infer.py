'''
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
anchor_input (InputLayer)       [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
positive_input (InputLayer)     [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
negative_input (InputLayer)     [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
sequential (Sequential)         (None, 10)           36843978    anchor_input[0][0]               
                                                                 positive_input[0][0]             
                                                                 negative_input[0][0]             
__________________________________________________________________________________________________
merged_layer (Concatenate)      (None, 30)           0           sequential[1][0]                 
                                                                 sequential[2][0]                 
                                                                 sequential[3][0]                 
==================================================================================================
Total params: 36,843,978
Trainable params: 36,589,770
Non-trainable params: 254,208
__________________________________________________________________________________________________
None
'''


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import pairwise_distances
import random
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

def create_base_network():
    """
    Base network to be shared.
    ## input_dim = (224, 224. 3)
    """
    model = load_model('cifar_resnet_model_v2.h5')    
    return model

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
    plt.savefig("TestTestImages" + str(len(reference_images)))

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

    
if __name__ == "__main__":
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
    encoder_network.load_weights('cifar_tsne_wts.h5', by_name = 'sequential')

    #use_case_one(encoder_network)
    use_case_two(encoder_network)