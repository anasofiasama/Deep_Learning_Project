# System --------------------------------------------------------
import os
import pathlib
# Dataframes and matrices ---------------------------------------
import numpy as np
import pandas as pd
# Graphics ------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # para imagenes
# Machine learning ----------------------------------------------
from sklearn.model_selection import train_test_split
# Deep learning -------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # modelo secuencial que toma paso por paso para generar la arq
from tensorflow.keras.layers import Conv2D,Dense,Dropout, Flatten # Conv2D: Convolusion y centralizacion de la imagen. Dropout: La red no se memoriza la red entera.
# Flatten estira la matriz.
from tensorflow.keras.layers import Activation, BatchNormalization #Funcion de activacion
from tensorflow.keras.layers import MaxPooling2D 
from tensorflow.keras import datasets, layers, models
from keras.utils import load_img 
from keras.utils import img_to_array
from keras.utils import get_file
from keras.utils import image_dataset_from_directory
# Save model
import pickle

#unzip files
!unzip ../data/raw/Cat.zip
!mv ResizedCat Cat

!unzip ../data/raw/Dog.zip
!mv ResizedDog Dog

# Create train dataset
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
BATCH_SIZE = 32 

train_ds = tf.keras.utils.image_dataset_from_directory( # image dataset from directory nombra cada elemento con el nombre de la carpeta
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
  batch_size=BATCH_SIZE)

  # Create validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
  batch_size=BATCH_SIZE)

# Create the deep learning architecture and fit the model
IMAGE_CHANNELS=3 # las imagenes a color tienen 3 canales: Rojo, Verde y Azul (RGB)

model = Sequential([

# Capa 1
Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25), # tecnica para que la red no se aprenda exactamente los datos, elimino el 25%

Conv2D(64, (3, 3), activation='relu'),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),

Conv2D(128, (3, 3), activation='relu'),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),

# ultima capa
Flatten(),
Dense(512, activation='relu'),
BatchNormalization(),
Dropout(0.5),
Dense(1, activation='sigmoid'), # 2 because we have cat and dog classes
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history=model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save the model as a pickle
filename = '../models/claf_model.pkl'
pickle.dump(model, open(filename,'wb'))

# Prediction
!unzip ../data/raw/Pred.zip
!mv ResizedPred Pred

train_ds = tf.keras.utils.image_dataset_from_directory( # image dataset from directory nombra cada elemento con el nombre de la carpeta
  test_dir,
  seed=123,
  image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
  batch_size=BATCH_SIZE)

pred=model.predict(train_ds)
