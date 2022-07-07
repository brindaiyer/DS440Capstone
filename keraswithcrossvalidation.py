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
from pylab import *
from sklearn.manifold import TSNE
from keras.layers import Input
from keras.layers import Activation, Dense
from keras import regularizers
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


#flattening data, appending class to features
data = pd.read_csv('C:/Users/brind/Documents/startingdata.csv')

# Trim the data to include only labeled data.
#old_data = data[data['result'] == "unknown"]
#pd.to_numeric(old_data['result'].replace({"unknown": "-1"}, inplace=True))
new_data = data[data['result'] != "unknown"]
new_data['result'] = pd.to_numeric(new_data.result) - 1

# Split 80/20 to train/test data
msk = np.random.rand(len(new_data)) < 0.8

x_train = new_data[msk]
x_train.pop('txId')

x_test = new_data[~msk]
predictionids = x_test.pop('txId')

#y_train = x_train.pop('result')
y_test = x_test.pop('result')

fraud = x_train[x_train['result'] == 1]#majority class
nonfraud = x_train[x_train['result'] == 0]#minority class
nonfraud = nonfraud.sample(len(fraud),replace=True)

x_train = pd.concat([nonfraud,fraud], axis=0)
y_train = x_train.pop('result')

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# Merge inputs and targets
Xfiles = np.concatenate((x_train, x_test), axis=0)
yfiles = np.concatenate((y_train, y_test), axis=0)

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold = []
loss_per_fold = []
for train, test in kfold.split(Xfiles, yfiles):

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
    #model.fit(x_train, y_train, epochs = 5) #epochs stands for how many times you go through training set

    # Generate generalization metrics
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
  
    ### Make predictions
    ### A Softmax layer converts the Logit output to probabilities
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    ##
    ### Use the model to predict the class of the test data
    predictions = probability_model.predict(x_test)

    ##### Compute and show the MCC
    incorrect = 0
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    predicted = []

    pred = np.argmax(predictions,axis=1)
    #print(pred)
    

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

    # Increase fold number
    fold_no = fold_no + 1

print("Validation accuracy:", sum(acc_per_fold)/len(acc_per_fold))
submission = pd.DataFrame({
        "txId": [predictionids.iloc[i] for i in range(len(predictionids))],
        "Class": pred 
    })

#submission.to_csv(r'C:\Users\brind\Documents\DS440\EvolveGCN\data\elliptic_bitcoin_dataset\KerasSequentialModels.csv', index = None, header=True)

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

dictionary = {'y_Actual': y_test, 'y_Predicted': predicted}

df = pd.DataFrame(dictionary)

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

sns.heatmap(confusion_matrix, annot=True)
plt.show()
