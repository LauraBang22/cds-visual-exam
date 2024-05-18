import os
import sys
sys.path.append("..")
import cv2

# Import teaching utils
import numpy as np
from imutils import jimshow as show 
from imutils import jimshow_channel as show_channel

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

# Import dataset
from tensorflow.keras.datasets import cifar10

# numpy
import numpy as np

# from scikit learn
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# TensorFlow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import SGD, Adam

# scikeras wrapper
from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm

def load_data():
    """Loads the CIFAR-10 dataset and reshapes labels."""
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.ravel()  # Reshape y_train to 1D
    y_test = y_test.ravel()    # Reshape y_test to 1D
    return X_train, X_test, y_train, y_test

def reshape_data(X_train, X_test):
    X_list_train = []

    for image in X_train:
        X_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_normalized = X_grey/255
        X_list_train.append(X_normalized)

    X_train_final = np.array(X_list_train).reshape(-1, 1024)

    X_list_test = []

    for image in X_test:
        X_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X_normalized = X_grey/255
        X_list_test.append(X_normalized)

    X_test_final = np.array(X_list_test).reshape(-1, 1024)
    return X_test_final, X_train_final

def classifier(X_test_final, X_train_final, y_test, y_train):
    param_grid = {
        'activation': ['relu'],
        'hidden_layer_sizes': [(75,), (100,), (125)],
        'max_iter': [200, 300]
    }

    classifier = MLPClassifier(random_state=42)

    grid_search = GridSearchCV(classifier, 
                                param_grid, 
                                cv=5, 
                                n_jobs=-1, 
                                verbose=2)
    grid_search.fit(X_train_final, y_train)
    
    best_classifier = grid_search.best_estimator_
    y_pred = best_classifier.predict(X_test_final)

    classifier_metrics_neural = metrics.classification_report(y_test, y_pred, target_names=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    
    print("Best parameters found by GridSearchCV:")
    print(grid_search.best_params_)
    
    print(classifier_metrics_neural)
    
    return classifier_metrics_neural, best_classifier

def main():
    X_train, X_test, y_train, y_test = load_data() 
    X_test_final, X_train_final = reshape_data(X_train, X_test) 
    classifier_metrics_neural, best_classifier = classifier(X_test_final, X_train_final, y_test, y_train)

if __name__=="__main__":
    main()