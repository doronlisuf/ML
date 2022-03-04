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

# Save the test input data and output data into data_train.npy and labels_train.npy
# Loading Data. replace 'data_train.npy' with 'data_test.npy' and 'labels_train.npy' with 'labels_test.npy'

def set_var(data_test_file, labels_test_file):
    data_test = np.load(data_test_file)
    labels_test = np.load(labels_test_file)

    labels_names =['Stadium','Building','Traffic Sign','Forest','Flowers',
                  'Street','Classroom','Bridge','Statue','Lake']
    return data_test, labels_test, labels_names

data_test, labels_test, labels_names = set_var('data_train.npy', 'labels_train.npy')

# loading the pre-trained model
from keras.models import load_model

def loading_model():
    y_final_new = labels_test - 1 #to have 0-9 instead of 1-10
    number_of_images = labels_test.shape[0]
    data_test_resized = np.resize(data_test.transpose(),(number_of_images,300,300,3))

    #load
    model_new = tf.keras.models.load_model('final_model.h5')

    predictions = model_new.predict(data_test_resized)
    scores_cnn = tf.nn.softmax(predictions)
    scores_cnn_np = np.array(scores_cnn)

    #predicting the classifications
    final_scores = scores_cnn_np
    print(final_scores)
    num_pred = final_scores.shape[0]
    final_predictions = np.zeros(num_pred)
    #final_scores.shape
    for i in range(num_pred):
        index = np.where(final_scores[i,:] == np.amax(final_scores[i,:]))
        final_predictions[i] = index[0] + 1

final_predictions = loading_model()
print(final_predictions)

conf_matrix_boost = confusion_matrix(labels_test, final_predictions)
class_report_boost = classification_report(labels_test, final_predictions)
print(conf_matrix_boost)
print(class_report_boost)