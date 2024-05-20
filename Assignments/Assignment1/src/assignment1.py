import os
import sys
sys.path.append("..")

import cv2 

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

def reference_image():
    '''
    This function defines the image I want, to compare all the others to 
    and find the most similar photo to.
    It reurns the image as a numpy array.
    '''
    image = os.path.join("in", "17flowers", "jpg", "image_0175.jpg")
    main_image = cv2.imread(image)
    return main_image

def normalize_image(main_image):
    '''
    This function normalizes the main image by creating a histogram.
    It returns a normalized histogram.
    '''
    main_hist = cv2.calcHist([main_image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    main_norm_hist = cv2.normalize(main_hist, main_hist, 0, 1.0, cv2.NORM_MINMAX)
    return main_norm_hist

def most_similar(main_norm_hist):
    '''
    This function finds the most similar photos to the main photo.
    It does so by iterating through all the photos in the in folder,
    and comparing them with the main photo
    It returns a datafram of the six most similar photos, and the path to the folder with the data.
    '''
    #pathway to the folder with all the flower photos
    data_dir = os.path.join("in", "17flowers", "jpg")
    sorted_dir = sorted(os.listdir(data_dir))

    #a list, that the results will be added to at a later time
    results_list = []

    #a for loop to treat each individual flower image in the data directory
    for file_name in sorted_dir:
        file_path = data_dir + "/" + file_name
        #if os.path.splitext(file_path)[1].lower() == ".jpg":
        ref_image = cv2.imread(file_path)
        
        ref_hist = cv2.calcHist([ref_image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
        ref_norm_hist = cv2.normalize(ref_hist, ref_hist, 0, 1.0, cv2.NORM_MINMAX)
        
        distance = round(cv2.compareHist(main_norm_hist, ref_norm_hist, cv2.HISTCMP_CHISQR), 2)
        
        results_list.append((file_name, distance))
        
        df = pd.DataFrame(results_list, 
                            columns=["file_name","Distance"],)
        df_sorted = df.sort_values(by='Distance', ascending=True)
        df_top = df_sorted.head(6)
    return df_top, data_dir

def save_results(df_top):
    '''
    This function saves the results as an dataframe
    '''
    outpath = os.path.join("out", "results_list_top.csv")
    df_top.to_csv(outpath, index=False)

def show_results(df_top, data_dir):
    '''
    This function plots and saves my results, by displaying the three most similar photos.
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

    plt.savefig("out/result.png")

def main():
    main_image = reference_image()
    main_norm_hist = normalize_image(main_image)
    df_top, data_dir = most_similar(main_norm_hist)
    save_results(df_top)
    show_results(df_top, data_dir)

if __name__ == "__main__":
    main()
