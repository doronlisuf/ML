import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
%matplotlib inline
plt.style.use('bmh')

#!pip install tensorflow
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Loading Data
data_train = np.load('data_train.npy')
labels_train = np.load('labels_train.npy')

labels_names =['Stadium','Building','Traffic Sign','Forest','Flowers',
              'Street','Classroom','Bridge','Statue','Lake']

print(data_train.shape, labels_train.shape)

X_train, X_test, y_train, y_test = train_test_split(data_train.transpose(), labels_train, test_size=0.20)

X_train, X_test, y_train, y_test
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#X_train = np.resize(X_train,(2499,300,300,3))
#X_test = np.resize(X_test,(625,300,300,3))
#y_train_new = y_train - 1
#y_test_new = y_test - 1
#print(y_train_new)
#train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_new))
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_new))

# keras imports for the dataset and building our neural network
#from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

# loading the dataset
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # building the input vector from the 32x32 pixels
X_train = X_train.reshape(X_train.shape[0], 300, 300, 3)
X_test = X_test.reshape(X_test.shape[0], 300, 300, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train - 1
y_test= y_test - 1

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# # building a linear stack of layers with the sequential model
# model = Sequential()

# # convolutional layer
# model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(300, 300, 3)))

# # convolutional layer
# model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# # flatten output of conv
# model.add(Flatten())

# # hidden layer
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(250, activation='relu'))
# model.add(Dropout(0.3))
# # output layer
# model.add(Dense(10, activation='softmax'))

# # compiling the sequential model
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# # training the model for 10 epochs
# model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = 10

model = Sequential([
  layers.Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(300, 300, 3)),
  layers.Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
  layers.MaxPool2D(pool_size=(2,2)),
  layers.Dropout(0.25),
  layers.Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
  layers.MaxPool2D(pool_size=(2,2)),
  layers.Dropout(0.25),  
  layers.Flatten(),
  layers.Dense(500, activation='relu'),
  layers.Dropout(0.4),
  layers.Dense(250, activation='relu'),
  layers.Dropout(0.3),
  layers.Dense(10, activation='softmax')
])

# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(300,300,3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

model.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
               
epochs=10
history = model.fit(
  train_dataset,
  validation_data=test_dataset,
  epochs=epochs
)               