# install libraries
!pip install opencv-python
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy.random as npr
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
%matplotlib inline
plt.style.use('bmh')

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras

# Loading Data
# Save input dataset X into data_train.npy and desired output vector Y into labels_train.npy
# Make sure data_train.npy and labels_train.npy are in the same directory

def set_variables(data_train_file, labels_train_file):
    data_train = np.load(data_train_file)
    labels_train = np.load(labels_train_file)

    labels_names =['Stadium','Building','Traffic Sign','Forest','Flowers',
              'Street','Classroom','Bridge','Statue','Lake']

    return data_train, labels_train, labels_names

data_train, labels_train, labels_names = set_variables('data_train.npy', 'labels_train.npy')


# training the CNN Model

def setup_model(batch_size, test_size_param):
    # splitting for training data and validation data
    X_train, X_test, y_train, y_test = train_test_split(data_train.transpose(), labels_train, test_size=test_size_param)
    X_train, X_test, y_train, y_test
    X_train = np.resize(X_train, (X_train.shape[0], 300, 300, 3))
    X_test = np.resize(X_test, (X_test.shape[0], 300, 300, 3))
    y_train_new = y_train - 1
    y_test_new = y_test - 1
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_new))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_new))

    BATCH_SIZE = batch_size
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    image_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    # data augmentation
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(300,
                                           300,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    for images, _ in train_dataset.take(1):
        for i in range(25):
            augmented_images = data_augmentation(images)
    return train_dataset, test_dataset, data_augmentation


train_dataset, test_dataset, data_augmentation = setup_model(64, 0.10)

# train model
def train_model(num_epochs):
    num_classes = 10

    model_aug = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(128, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(256, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(512, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(1024, activation='relu'),
      layers.Dense(num_classes)
    ])

    model_aug.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # save model as final_model.h5. This pretrained model will be used in test_function.ipynb
    checkpoint_filepath = 'final_model.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    # set number of epochs
    epochs = num_epochs
    history = model_aug.fit(
      train_dataset,
      validation_data=test_dataset,
      epochs=epochs,
      callbacks = [model_checkpoint_callback]
    )

train_model(80)