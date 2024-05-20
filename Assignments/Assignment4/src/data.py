from facenet_pytorch import MTCNN
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
    return mtcnn

def run_model(mtcnn):
    '''
    This function iterates through scans of newspaper pages and 
    detects faces in each page, and counts the number of faces found using the model.
    It returns a list where each result is a tuple containing newspaper, year and number of faces
    '''
    main_folder_path = ("in/newspapers") #the folder that we will be working in
    sorted_dir = sorted(os.listdir(main_folder_path))

    results = []

    for newspaper in sorted_dir: #creating a "for loop" to reach all the subfolders
        print("Working on folder called: " + newspaper)
        folder_path = os.path.join(main_folder_path, newspaper) 
        filenames = sorted(os.listdir(folder_path)) 
        
        
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
    '''
    This function saves the results as a CSV fil in the out folder.
    '''
    data = pd.DataFrame(results, columns=["Newspaper","Year", "Faces"],)
    outpath = os.path.join("out", "data.csv")
    data.to_csv(outpath)

def main():
    mtcnn = load_model()
    results = run_model(mtcnn)
    save_data(results)


if __name__ == "__main__":
    main()
