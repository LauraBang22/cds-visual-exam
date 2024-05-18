import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    '''
    loading the model I'll use on the data
    '''
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(224, 224, 3))
    return model

def main_image(model):
    '''
    defining the image I want, to compare all the others to and find the most similar photo to.
    '''
    input_image = Image.open("in/flowers/image_0175.jpg")
    resized_image = input_image.resize((224, 224))
    image_array_main = np.expand_dims(img_to_array(resized_image), axis = 0)
    image_embedding_main = model.predict(image_array_main)
    return image_embedding_main

def most_similar(model, image_embedding_main):
    #pathway to the folder with all the flower photos
    data_dir = os.path.join("in", "flowers")
    sorted_dir = sorted(os.listdir(data_dir))

    #I create a list, that I will add my results to at a later time in the code
    results_list = []

    #Then I create a for loop to treat each individual flower image in the data directory
    for file_name in sorted_dir:
        file_path = data_dir + "/" + file_name
        resized_image = load_img(file_path, target_size=(224, 224))
        image_array = np.expand_dims(img_to_array(resized_image), axis = 0)
        image_embedding = model.predict(image_array)
        similarity_score = cosine_similarity(image_embedding_main, image_embedding).reshape(1,)
        results_list.append((file_name, similarity_score))
        df = pd.DataFrame(results_list, columns=["file_name", "similarity_score"]) 
        df_sorted = df.sort_values(by='similarity_score', ascending=False)
        df_top = df_sorted.head(6)
    return df_top, data_dir

def save_results(df_top):
    '''
    saving my results as an dataframe
    '''
    outpath = os.path.join("out", "results_list_top_vgg16.csv")
    df_top.to_csv(outpath, index=False)

def show_results(df_top, data_dir):
    '''
    plotting my results
    '''
    top_image_1_path = os.path.join(data_dir, df_top.iloc[0]['file_name'])
    top_image_2_path = os.path.join(data_dir, df_top.iloc[1]['file_name'])
    top_image_3_path = os.path.join(data_dir, df_top.iloc[2]['file_name'])

    top_image_1 = Image.open(top_image_1_path)
    top_image_2 = Image.open(top_image_2_path)
    top_image_3 = Image.open(top_image_3_path)

    top_image_1_array = np.array(top_image_1)
    top_image_2_array = np.array(top_image_2)
    top_image_3_array = np.array(top_image_3)

    
    plt.figure(figsize=(10, 6))  

    plt.subplot(1, 3, 1)  
    plt.imshow(top_image_1_array)
    plt.title(df_top.iloc[0]['file_name'])
    plt.axis('off') 
    
    plt.subplot(1, 3, 2)  
    plt.imshow(top_image_2_array)
    plt.title(df_top.iloc[1]['file_name'])
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(top_image_3_array)
    plt.title(df_top.iloc[2]['file_name'])
    plt.axis('off') 

    plt.savefig("out/result_vgg16.png")

def main():
    model = load_model()
    image_embedding_main = main_image(model)
    df_top, data_dir = most_similar(model, image_embedding_main)
    save_results(df_top)
    show_results(df_top, data_dir)

if __name__ == "__main__":
    main()

