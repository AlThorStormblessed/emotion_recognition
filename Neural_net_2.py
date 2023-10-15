import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

from PIL import Image
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import glob
import random
import os
import cv2
import time
import tensorflow as tf
from keras import optimizers, Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.layers import Input, Lambda, Dense, Flatten, Dropout,BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPool2D, Activation
from keras.models import Sequential
from keras.applications import MobileNet, ResNet50, xception
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from deepface import DeepFace
from deepface.detectors import FaceDetector


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

print("Imports done")
count = 0

Images_df = pd.read_csv("fer2013.csv")
X_train = [[int(pixel) for pixel in image.split()] for image in tqdm(Images_df[Images_df["Usage"] == "Training"]["pixels"].values)]
y_train = [emotion for emotion in Images_df[Images_df["Usage"] == "Training"]["emotion"].values]

y_train = to_categorical(y_train, dtype ="uint8")
X_train = np.array(X_train, dtype = 'float16').reshape(-1, 48 * 48 * 1)

X_test = [[int(pixel) for pixel in image.split()] for image in tqdm(Images_df[Images_df["Usage"] == "PublicTest"]["pixels"].values)]
y_test = [emotion for emotion in Images_df[Images_df["Usage"] == "PublicTest"]["emotion"].values]

X_test.extend([[int(pixel) for pixel in image.split()] for image in tqdm(Images_df[Images_df["Usage"] == "PrivateTest"]["pixels"].values)])
y_test.extend([emotion for emotion in Images_df[Images_df["Usage"] == "PrivateTest"]["emotion"].values])

y_test = to_categorical(y_test, dtype ="uint8")
X_test = np.array(X_test, dtype = 'float16').reshape(-1, 48 * 48 * 1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_train = X_train.reshape(-1, 48, 48, 1)

scaler.fit(X_test)
X_test = scaler.transform(X_test)

X_test = X_test.reshape(-1, 48, 48, 1)

def get_model():

    model=Sequential()

    model.add(Conv2D(64,(3,3),input_shape=(48,48,1), padding = "same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(2,2))
    # model.add(Dropout(0.25))

    model.add(Conv2D(128,(5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(2,2))
    # model.add(Dropout(0.25))

    model.add(Conv2D(256,(5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(2,2))
    # model.add(Dropout(0.25))

    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(MaxPool2D(2,2))

    model.add(GlobalAveragePooling2D())

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Dropout(0.25))
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    # model.add(Dropout(0.25))

    #adding output layer
    model.add(Dense(7,activation='softmax'))
    
    return model

model = get_model()
model.compile(optimizer= Adam(learning_rate=0.01),
              loss = "categorical_crossentropy",
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    horizontal_flip = True,
    rotation_range=40
)

checkpointer = ModelCheckpoint('saved_model/model_t', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.000001, mode='auto')
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

data = datagen.flow(X_train, y_train, batch_size = 64)
validation_data = datagen.flow(X_test, y_test, batch_size = 64)

history = model.fit(data, batch_size=64, 
          validation_data=validation_data,
          callbacks = [checkpointer, earlystopper, reduce_lr],
          epochs=80)
model.save("saved_model/model_t")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.grid()
plt.show()