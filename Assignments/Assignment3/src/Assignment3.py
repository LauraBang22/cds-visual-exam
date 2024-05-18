import os
import cv2
import tensorflow as tf

from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


labelNames = ["ADVE", "Email", 
              "Form", "Letter", 
              "Memo", "News", 
              "Note", "Report", 
              "Resume", "Scientific"]

def load_data():
    '''
    Loading the data, and giving the images a label each
    '''
    main_folder_path = ("in/Tobacco3482-jpg") 
    sorted_dir = sorted(os.listdir(main_folder_path))

    images = []
    labels = []
    for folder in sorted_dir:
        label = folder.split("-")[-1] # extract label from folder name
        folder_path = os.path.join(main_folder_path, folder)
        filenames = sorted(os.listdir(folder_path))
        
        for image in filenames:
            if image.endswith(".jpg"):
                image_path = os.path.join(folder_path, image)
                labels.append(label)
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                images.append(image)
    return images, labels

def reshape_data(images, labels):
    '''
    Reshaping and normalizing the data, so it fits with the model
    '''
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    X_train = np.array(X_train) / 255.
    X_test = np.array(X_test) / 255.

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    return X_train, X_test, y_train, y_test


def load_model():
    '''
    loading the model and adding some layers to it.
    '''
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False

    flat1 = Flatten()(model.layers[-1].output)
    bn=BatchNormalization()(flat1)
    class1 = Dense(128, activation='relu')(bn)
    output = Dense(10, activation='softmax')(class1)

    model = Model(inputs=model.inputs, 
                outputs=output)
    
    return model

def learning_rate():
    '''
    defining the learning rate of the neural network
    '''
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    return sgd

def compile_model(sgd, model):
    '''
    compiling the model, so the learning rate is added to it
    '''
    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train):
    '''
    train the model on the data
    '''
    H = model.fit(X_train, y_train, 
                validation_split=0.1,
                batch_size=128,
                epochs=25,
                verbose=1)
    return H

def plot_history(H):
    '''
    plotting the loss curve and accuracy curve over the trained model.
    the plots also include validation loss and accuracy to give an indication
    of whether the model is overfitting or underfitting.
    '''
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, 25), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 25), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, 25), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 25), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig("out/loss_curve_test.png")

def predictions(model, X_test, y_test, labelNames):
    '''
    making the classification report based on predictions
    '''
    predictions = model.predict(X_test, batch_size=128)
    classification = classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames)
    return classification

def file_save(classification):
    with open('out/classification.txt', 'w') as text_file:
        text_file.write(classification)

def main():
    images, labels = load_data()
    X_train, X_test, y_train, y_test = reshape_data(images, labels)
    
    model = load_model()
    sgd = learning_rate()
    model = compile_model(sgd, model)
    
    H = train_model(model, X_train, y_train)
    
    plot_history(H)
    
    classification = predictions(model, X_test, y_test, labelNames)
    file_save(classification)

if __name__ == "__main__":
    main()

