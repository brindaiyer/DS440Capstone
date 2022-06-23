import numpy as np
import time
import sys
import pandas as pd
import os
import seaborn as sns
import math
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from pylab import *
from sklearn.manifold import TSNE
from keras.layers import Input
from keras.layers import Activation, Dense
from keras import regularizers
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight


data = pd.read_csv('C:/Users/brind/Documents/startingdata.csv')

# Trim the data to include only labeled data.
old_data = data[data['result'] == "unknown"]
new_data = data[data['result'] != "unknown"]
new_data['result'] = pd.to_numeric(new_data.result) - 1

# Split 80/20 to train/test data
msk = np.random.rand(len(new_data)) < 0.8

x_train = new_data[msk]
x_train.pop('txId')

x_test = new_data[~msk]
predictionid = x_test.pop('txId')
print(predictionid)

y_test = x_test.pop('result')

fraud = x_train[x_train['result'] == 1]#majority class
nonfraud = x_train[x_train['result'] == 0]#minority class
nonfraud = nonfraud.sample(len(fraud),replace=True)
####pd.concat
####
##
x_train = pd.concat([nonfraud,fraud], axis=0)
y_train = x_train.pop('result')


# build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(166),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



####
##### train the model
model.fit(x_train, y_train, epochs = 5) #epochs stands for how many times you go through training set

#saved_model = pickle.dumps(model)
####
### test the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
print('\nTest accuracy:', test_acc)
##
### Make predictions
### A Softmax layer converts the Logit output to probabilities
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

##
### Use the model to predict the class of the test data
predictions = probability_model.predict(x_test)

####
##### Compute and show the MCC
incorrect = 0
FP = 0
FN = 0
TP = 0
TN = 0
predicted = []

for i in range(len(y_test)):
    pred = np.argmax(predictions[i])
    predicted.append(pred)
    if pred == 0 and y_test.iloc[i] == 1:
        FN += 1
    
    if pred == 1 and y_test.iloc[i] == 0:
        FP += 1

    if pred == 0 and y_test.iloc[i] == 0:
        TN += 1

    if pred == 1 and y_test.iloc[i] == 1:
        TP += 1
        
    if pred != y_test.iloc[i]:
        incorrect +=1

submission = pd.DataFrame({
        "txId": [predictionid.iloc[i] for i in range(len(predictionid))],#x_test["timestep"],
        "Class": pred
    })

##
##submission.to_csv(r'C:\Users\brind\Documents\DS440\EvolveGCN\data\elliptic_bitcoin_dataset\KerasSequentialModels.csv', index = None, header=True)

print("Test Accuracy:", round(100 * (TP + TN) / len(y_test), 2), "%")
print("Only", incorrect, "wrong out of", len(y_test), ", or", round(100 * float(incorrect) / len(y_test), 2), "%\n")

print("Confusion Matrix")
print("----------------")
print("TN(",TN, ")\tFP(", FP, ")")
print("FN(",FN, ")\tTP(", TP, ")\n")

print("Matthews Correlation Coefficient (MCC)")
print("--------------------------------------")
MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))
print("MCC = ", round(MCC, 2), "on scale of [-1 1]")

dict = {'y_Actual': y_test, 'y_Predicted': predicted}

df = pd.DataFrame(dict)

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

sns.heatmap(confusion_matrix, annot=True)
plt.show()
