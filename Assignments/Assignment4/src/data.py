from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import cv2

def load_model():
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    return mtcnn, resnet

def run_model(mtcnn, resnet):
    main_folder_path = ("in/newspapers") #the folder that we will be working in
    sorted_dir = sorted(os.listdir(main_folder_path)) #sorting all the subfolders
    results = []

    for newspaper in sorted_dir: #creating a "for loop" to reach all the subfolders
        print("Working on folder called: " + newspaper)
        folder_path = os.path.join(main_folder_path, newspaper) 
        filenames = sorted(os.listdir(folder_path)) #sorting all the files in the different subfolders
        #Define a empty list for later use
        
        for filename in tqdm(filenames): #creating a new "for loop" to reach all the files in the subfolders
            filepath = folder_path + "/" + filename
            year = filename[4:8]        
            
            img = Image.open(filepath)
            boxes, _ = mtcnn.detect(img)

            if boxes is not None: 
                results.append((newspaper, year, len(boxes)))
            else:
                results.append((newspaper, year, 0))
            print("Through"+filename)
    return results

def save_data(results):
    data = pd.DataFrame(results, columns=["Newspaper","Year", "Faces"],)
    outpath = os.path.join("out", "data.csv")
    data.to_csv(outpath)

def main():
    mtcnn, resnet = load_model()
    results = run_model(mtcnn, resnet)
    save_data(results)


if __name__ == "__main__":
    main()
