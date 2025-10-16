#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 22:21:16 2025

@author: bradsommer

This code is adapted from CS 435 Module 4 Exercise 1 & 2 sample code written by
Dr. G.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------------
# Step 1.1 - Read the dataset into python
# ----------------------------------------
filename = "./Mod4_SummerStudentAdmissionsData_3Labeled_4D.csv"
DF = pd.read_csv(filename)
print(DF)


# ----------------------------------------
# Step 1.2 - One-hot encode the label
# ----------------------------------------
# Print each part to see what is going on
print("Labels in DF:\n", DF.iloc[:, 0])
Labels = np.array(DF.iloc[:, 0])
print("Extracted lables as an (n,) 1D array:\n", Labels)

# Encode labels as 0 1 2
MyLabelEncoder = LabelEncoder()
Encoded_labels = MyLabelEncoder.fit_transform(Labels)
print("Encoded labels:\n", Encoded_labels)

# One-hot encoding
MyOneHotEncoder = OneHotEncoder()
Encoded_labels = Encoded_labels.reshape(-1, 1)
print("Encoded labels with shape(n, 1):\n", Encoded_labels)
OneHotEncodedLabels = MyOneHotEncoder.fit_transform(Encoded_labels).toarray()
print("One-hot encoded lables:\n", OneHotEncodedLabels)


# ----------------------------------------
# Step 1.3. - Save label separately
# ----------------------------------------
# the original labels are saved as Label
print("The original labels as found in the dataset and saved separately:\n",
      Labels)
# Final one-hot encoded lables saved separately as well
HotLabels = OneHotEncodedLabels
print("Final one-hot encoded lables are separated as well\n", HotLabels)


# ----------------------------------------
# 1.4 - Save the data separately
# ----------------------------------------
MyData = DF.drop(columns="Decision", axis=1)
print("Data saved separately from the labels:\n", MyData)


# ----------------------------------------
# 2.1 - Split data into training and testing data
# ----------------------------------------
TrainDF, TestDF, TrainLabels, TestLabels = train_test_split(MyData, HotLabels, test_size=0.3, random_state=13)
# Training data and labels
print("Training dataset:\n", TrainDF)
print("Training labels:\n", TrainLabels)

# Testing data and labels
print("Testing dataset:\n", TestDF)
print("Testing labels:\n", TestLabels)

# ----------------------------------------
# 2.2 - Normalize data
# ----------------------------------------
MyMM = MinMaxScaler()

TrainDF = MyMM.fit_transform(TrainDF)
print("Normalized training dataset:\n", TrainDF)

TestDF = MyMM.fit_transform(TestDF)
print("Normalized testing dataset\n", TestDF)

# ----------------------------------------
# 3.1 - Instantiate a tf.keras Sequential Model
# ----------------------------------------
My_NN_Model = tf.keras.models.Sequential([
    # The input shape of our data is a vector of 4 values.
    # Here, we are sending our data to a hidden layer that has 5 hidden units
    tf.keras.layers.Dense(5, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

My_NN_Model.summary()

# Print the parameters - weights and biases
first_layer_weights = My_NN_Model.layers[0].get_weights()[0]
first_layer_biases = My_NN_Model.layers[0].get_weights()[1]

second_layer_weights = My_NN_Model.layers[1].get_weights()[0]
second_layer_biases = My_NN_Model.layers[1].get_weights()[1]

third_layer_weights = My_NN_Model.layers[2].get_weights()[0]
third_layer_biases = My_NN_Model.layers[2].get_weights()[1]

fourth_layer_weights = My_NN_Model.layers[3].get_weights()[0]
fourth_layer_biases = My_NN_Model.layers[3].get_weights()[1]

fifth_layer_weights = My_NN_Model.layers[4].get_weights()[0]
fifth_layer_biases = My_NN_Model.layers[4].get_weights()[1]

print("The first layer weights are: \n", first_layer_weights)
print("The first layer biases are: \n", first_layer_biases)

print("The second layer weights are: \n", second_layer_weights)
print("The second layer biases are: \n", second_layer_biases)

print("The third layer weights are: \n", third_layer_weights)
print("The third layer biases are: \n", third_layer_biases)

print("The fourth layer weights are: \n", fourth_layer_weights)
print("The fourth layer biases are: \n", fourth_layer_biases)

print("The fifth layer weights are: \n", fifth_layer_weights)
print("The fifth layer biases are: \n", fifth_layer_biases)
