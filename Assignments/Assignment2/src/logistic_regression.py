import os
import sys
sys.path.append("..")
import cv2
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressio
from sklearn import metrics

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10


def load_data():
    return cifar10.load_data()

def reshape_data(X_train, X_test):
    '''
    reshaping the data, for the classifier to be able to run on it
    '''
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
    '''
    training the classifier on the data and creeating a classification report
    '''
    classifierLogistic = LogisticRegression(tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_final, y_train)
    y_pred = classifierLogistic.predict(X_test_final)
    classifier_metrics_logistic = metrics.classification_report(y_test, y_pred, target_names= ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    print(classifier_metrics_logistic)
    return classifier_metrics_logistic

def file_save(classifier_metrics_logistic):
    text_file = open("out/logisticregression.txt", 'w')
    text_file.write(classifier_metrics_logistic)
    text_file.close()


def main():
    (X_train, y_train), (X_test, y_test) = load_data()
    X_test_final, X_train_final = reshape_data(X_train, X_test)
    classifier_metrics_logistic = classifier(X_test_final, X_train_final, y_test, y_train)
    file_save(classifier_metrics_logistic)

if __name__=="__main__":
    main()