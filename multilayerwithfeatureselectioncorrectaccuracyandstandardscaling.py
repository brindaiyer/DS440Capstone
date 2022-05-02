import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import networkx as nx
from tensorflow.python.framework import graph_util
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import torch
from torch import nn
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow import keras
from matplotlib import pyplot
from feature_engine.selection import SelectBySingleFeaturePerformance

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import seaborn as sns
from sklearn.manifold import TSNE
import imblearn
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn import preprocessing 
from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve,roc_auc_score,fbeta_score,recall_score
from sklearn.metrics import plot_precision_recall_curve,average_precision_score,f1_score
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Read in and pre-process data
id_time=["txId", "time_step"]
columns = ['txId', 'timestep']

class_names = ['Illicit', 'Legal']

# Name the columns without known names
for x in range(165) :
    columns.append('col'+ str(x))
feature_names = ['feature_'+str(i) for i in range(1,166)]
column_names = id_time + feature_names
classes_csv = pd.read_csv('C:/Users/brind/Documents/DS440/archive/elliptic_bitcoin_dataset/elliptic_txs_classes.csv') #txId, class
features_csv = pd.read_csv('C:/Users/brind/Documents/DS440/archive/elliptic_bitcoin_dataset/elliptic_txs_features.csv',names=column_names)

# Flatten the data, append class to features
data = features_csv.assign(result=classes_csv['class'])

# Trim the data to include only labeled data. Fraud is 2, nonfraud is 1
test_file = data[data['result'] == "unknown"]
result = test_file['result']
test_file = test_file.loc[:,~test_file.columns.str.contains('result')]


train_file = data[data['result'] != "unknown"]
train_file['result'] = pd.to_numeric(train_file.result) - 1 #fraud 1, nonfraud 0



fraud = train_file[train_file['result'] == 1]
nonfraud = train_file[train_file['result'] == 0]
fraudsample = fraud.sample(n=len(nonfraud),random_state=1)
train_file = pd.concat([fraudsample, nonfraud], axis=0)
ytrain_file = train_file.pop('result')

dict_class = { 
    0 : 'Non-fraud',
    1 : 'Fraud'
}

sel = SelectBySingleFeaturePerformance(estimator=LinearRegression(), scoring="r2", cv=3, threshold=0.01)
sel.fit(train_file,ytrain_file)
sel.features_to_drop_
sel.feature_performance_
train_file_features = sel.transform(train_file)

features = train_file_features.to_numpy()
labels = ytrain_file.to_numpy()

# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    import numpy as np
    
    if (der==True) : #derivative of the sigmoid
        f = x/(1-x)
    else : # sigmoid
        f = 1/(1+ np.exp(-x))
    
    return f

# We may employ the Rectifier Linear Unit (ReLU)
def ReLU_act(x, der=False):
    import numpy as np
    
    if (der== True):
        if x>0 :
            f= 1
        else :
            f = 0
    else :
        if x>0:
            f = x
        else :
            f = 0
    return f

def perceptron(X, act='Sigmoid'): 
    import numpy as np
    
    shapes = X.shape # Pick the number of (rows, columns)!
    n= shapes[0]+shapes[1]
    # Generating random weights and bias
    w = 2*np.random.random(shapes) - 0.5 # We want w to be between -1 and 1
    b = np.random.random(1)
    
    # Initialize the function
    f = b[0]
    for i in range(0, X.shape[0]-1) : # run over column elements
        for j in range(0, X.shape[1]-1) : # run over rows elements
            f += w[i, j]*X[i,j]/n
    # Pass it to the activation function and return it as an output
    if act == 'Sigmoid':
        output = sigmoid_act(f)
    else :
        output = ReLU_act(f)
        
    return output

#output of perceptron
print('Output with sigmoid activator: ', perceptron(features))
print('Output with ReLU activator: ', perceptron(features))

#perceptron
# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    import numpy as np
    
    if (der==True) : #derivative of the sigmoid
        f = x/(1-x)
    else : # sigmoid
        f = 1/(1+ np.exp(-x))
    
    return f

# We may employ the Rectifier Linear Unit (ReLU)
def ReLU_act(x, der=False):
    import numpy as np
    
    if (der== True):
        if x>0 :
            f= 1
        else :
            f = 0
    else :
        if x>0:
            f = x
        else :
            f = 0
    return f

# Now we are ready to define the perceptron; 
# it eats a np.array (that may be a list of features )
def perceptron(X, act='Sigmoid'): 
    import numpy as np
    
    shapes = X.shape # Pick the number of (rows, columns)!
    n= shapes[0]+shapes[1]
    # Generating random weights and bias
    w = 2*np.random.random(shapes) - 0.5 # We want w to be between -1 and 1
    b = np.random.random(1)
    
    # Initialize the function
    f = b[0]
    for i in range(0, X.shape[0]-1) : # run over column elements
        for j in range(0, X.shape[1]-1) : # run over rows elements
            f += w[i, j]*X[i,j]/n
    # Pass it to the activation function and return it as an output
    if act == 'Sigmoid':
        output = sigmoid_act(f)
    else :
        output = ReLU_act(f)
    return output

#example of output of Perceptron
print('Output with sigmoid activator: ', perceptron(features))
print('Output with ReLU activator: ', perceptron(features))

#Neural Network layer
# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    import numpy as np
    
    if (der==True) : #derivative of the sigmoid
        f = 1/(1+ np.exp(- x))*(1-1/(1+ np.exp(- x)))
    else : # sigmoid
        f = 1/(1+ np.exp(- x))
    
    return f

# We may employ the Rectifier Linear Unit (ReLU)
def ReLU_act(x, der=True):
    import numpy as np
    
    if (der == True): # the derivative of the ReLU is the Heaviside Theta
        f = np.heaviside(x, 1)
    else :
        f = np.maximum(x, 0)
    
    return f

#putting labels and features into a train and test set
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.20, shuffle=True,random_state=1)
#X_train, X_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, shuffle=True,random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print('Training records:',Y_train.size)
print('Test records:',Y_test.size) #

#training the model
# Set up the number of perceptron per each layer:
#original p=4,q=4
p=4 # Layer 1
q=4 # Layer 2

# Set up the Learning rate
eta =  1/623


# 0: Random initialize the relevant data 
w1 = 2*np.random.rand(p , X_train.shape[1]) - 0.5 # Layer 1
b1 = np.random.rand(p)

w2 = 2*np.random.rand(q , p) - 0.5  # Layer 2
b2 = np.random.rand(q)

wOut = 2*np.random.rand(q) - 0.5  # Output Layer
bOut = np.random.rand(1)

mu = []
vec_y = []

# Start looping over the the ids, i.e. over I.

for I in range(0, X_train.shape[0]): #loop in all the ids:
    
    # 1: input the data 
    x = X_train[I]
    
    
    # 2: Start the algorithm
    
    # 2.1: Feed forward
    z1 = ReLU_act(np.dot(w1, x) + b1) # output layer 1 
    z2 = ReLU_act(np.dot(w2, z1) + b2) # output layer 2
    y = sigmoid_act(np.dot(wOut, z2) + bOut) # Output of the Output layer

#2.2: Compute the output layer's error
    delta_Out =  (y-Y_train[I]) * sigmoid_act(y, der=True)
    
    #2.3: Backpropagate
    delta_2 = delta_Out * wOut * ReLU_act(z2, der=True) # Second Layer Error
    delta_1 = np.dot(delta_2, w2) * ReLU_act(z1, der=True) # First Layer Error
    
    # 3: Gradient descent 
    wOut = wOut - eta*delta_Out*z2  # Outer Layer
    bOut = bOut - eta*delta_Out
    
    w2 = w2 - eta*np.kron(delta_2, z1).reshape(q,p) # Hidden Layer 2
    b2 = b2 - eta*delta_2
    
    w1 = w1 - eta*np.kron(delta_1, x).reshape(p, x.shape[0]) # Hidden Layer 1

    b1 = b1 - eta*delta_1
    
    # 4. Computation of the loss function
    mu.append((1/2)*(y-Y_train[I])**2)
    vec_y.append(y[0])


# Plotting the Cost function for each training data     
plt.figure(figsize=(10,6))
plt.scatter(np.arange(0, X_train.shape[0]), mu, alpha=0.3, s=4, label='mu')
plt.title('Loss for each training data point', fontsize=20)
plt.xlabel('Training data', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.show()

# Plotting the average cost function over 10 training data    
pino = []
for i in range(0, 9):
    pippo = 0
    for m in range(0, 59):
        pippo+=vec_y[60*i+m]/60
    pino.append(pippo)
    
    

plt.figure(figsize=(10,6))
plt.scatter(np.arange(0, 9), pino, alpha=1, s=10, label='error')
plt.title('Averege Loss by epoch', fontsize=20)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.show()

#define training as a function
#function where gradient descent is done
def ANN_train(X_train, Y_train, p, q, eta):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 0: Random initialize the relevant data 
    w1 = 2*np.random.rand(p , X_train.shape[1]) - 0.5 # Layer 1
    b1 = np.random.rand(p)

    w2 = 2*np.random.rand(q , p) - 0.5  # Layer 2
    b2 = np.random.rand(q)

    wOut = 2*np.random.rand(q) - 0.5   # Output Layer
    bOut = np.random.rand(1)

    mu = []
    vec_y = []

    # Start looping over the bitcoin user ids, i.e. over I.

    for I in range(0, X_train.shape[0]-1): #(0,X_train.shape[0]) #loop in all the bitcoin user ids:
        # 1: input the data 
        x = X_train[I]
    
        # 2: Start the algorithm
    
        # 2.1: Feed forward
        z1 = ReLU_act(np.dot(w1, x) + b1) # output layer 1 
        z2 = ReLU_act(np.dot(w2, z1) + b2) # output layer 2
        y = sigmoid_act(np.dot(wOut, z2) + bOut) # Output of the Output layer
    
        #2.2: Compute the output layer's error
        delta_Out = 2 * (y-Y_train[I]) * sigmoid_act(y, der=True)
    
        #2.3: Backpropagate
        delta_2 = delta_Out * wOut * ReLU_act(z2, der=True) # Second Layer Error
        delta_1 = np.dot(delta_2, w2) * ReLU_act(z1, der=True) # First Layer Error

        # 3: Gradient descent 
        wOut = wOut - eta*delta_Out*z2  # Outer Layer
        bOut = bOut - eta*delta_Out
    
        w2 = w2 - eta*np.kron(delta_2, z1).reshape(q,p) # Hidden Layer 2
        b2 = b2 -  eta*delta_2
    
        w1 = w1 - eta*np.kron(delta_1, x).reshape(p, x.shape[0])
        b1 = b1 - eta*delta_1
    
        # 4. Computation of the loss function
        mu.append((y-Y_train[I])**2) 
        vec_y.append(y) #y[0]

    batch_loss = []
    for i in range(0, 10): #range (0,9)
        loss_avg = 0
        for m in range(0, 60): #range(0,59)
            loss_avg+=vec_y[60*i+m]/60
        batch_loss.append(loss_avg)

    plt.figure(figsize=(10,6))
    plt.scatter(np.arange(1, len(batch_loss)+1), batch_loss, alpha=1, s=10, label='error')
    plt.title('Averege Loss by epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()
    
    return w1, b1, w2, b2, wOut, bOut, mu

w1, b1, w2, b2, wOut, bOut, mu = ANN_train(X_train, Y_train, p=10, q=2, eta=0.0015)

X_train, X_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, shuffle=True,random_state=1)

#compute predictions from trained ANN
def ANN_pred(X_test, w1, b1, w2, b2, wOut, bOut, mu):
    import numpy as np
    
    pred = []
    
    for I in range(0, X_test.shape[0]): #loop in all the bitcoin ids
        # 1: input the data 
        x = X_test[I]
        
        # 2.1: Feed forward
        z1 = ReLU_act(np.dot(w1, x) + b1) # output layer 1 
        z2 = ReLU_act(np.dot(w2, z1) + b2) # output layer 2
        y = sigmoid_act(np.dot(wOut, z2) + bOut)  # Output of the Output layer
        
        # Append the prediction;
        # We now need a binary classifier; we this apply an Heaviside Theta and we set to 0.5 the threshold
        # if y < 0.5 the output is zero, otherwise is 1
        pred.append( np.heaviside(y - 0.5, 1)[0] )

    return np.array(pred);

predictions = ANN_pred(X_test, w1, b1, w2, b2, wOut, bOut, mu)

#Evaluation report
# Plot the confusion matrix
cm = confusion_matrix(Y_test, predictions)


df_cm = pd.DataFrame(cm, index = [dict_class[i] for i in range(0,2)], columns = [dict_class[i] for i in range(0,2)])
plt.figure(figsize = (7,7))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted Class", fontsize=18)
plt.ylabel("True Class", fontsize=18)
plt.show()


accuracy = round((100 * ((df_cm.iloc[1,1] + df_cm.iloc[0,0]) / (df_cm.iloc[1,1] + df_cm.iloc[0,0] + df_cm.iloc[0,1]+df_cm.iloc[1,0]))), 2)
print("Test Accuracy:", accuracy, "%") 

##
misclassification = ((df_cm.iloc[0,1]+df_cm.iloc[1,0])/(df_cm.iloc[1,1] + df_cm.iloc[0,0] + df_cm.iloc[0,1]+df_cm.iloc[1,0]))
precision = (df_cm.iloc[1,1])/(df_cm.iloc[1,1]+df_cm.iloc[0,1])
sensitivity = (df_cm.iloc[1,1])/(df_cm.iloc[1,1]+df_cm.iloc[1,0])
specificity = (df_cm.iloc[0,0])/(df_cm.iloc[0,0]+df_cm.iloc[0,1])#TN/TN+FP
print("Misclassification:", misclassification)
print("Precision:",precision)
print("Sensitivity:",sensitivity)
print("Specificity:",specificity)

print("Confusion Matrix")
print("----------------")
print("TN(",df_cm.iloc[0,0], ")\tFP(", df_cm.iloc[0,1], ")")
print("FN(",df_cm.iloc[1,0], ")\tTP(", df_cm.iloc[1,1], ")\n")

print("Matthews Correlation Coefficient (MCC)")
print("--------------------------------------")
MCC = ((df_cm.iloc[1,1] * df_cm.iloc[0,0]) - (df_cm.iloc[0,1] * df_cm.iloc[1,0])) / math.sqrt((df_cm.iloc[1,1] + df_cm.iloc[0,1]) * (df_cm.iloc[1,1] + df_cm.iloc[1,0]) * (df_cm.iloc[0,0] + df_cm.iloc[0,1] ) * (df_cm.iloc[0,0] + df_cm.iloc[1,0]))
print("MCC = ", round(MCC, 2), "on scale of [-1 1]")

# We apply the dictionary using a lambda function and the pandas .apply() module
X = test_file[list(train_file_features)].to_numpy()

test_predictions = ANN_pred(X, w1, b1, w2, b2, wOut, bOut, mu)

#exporting predictions to csv file
submission = pd.DataFrame({
        "txId": test_file["txId"],
        "Class": test_predictions
    })

print(submission)
# Export it in a 'Comma Separated Values' (CSV) file
#submission.to_csv(r'C:\Users\brind\Documents\DS440\firstattempt.csv', index = None, header=True)

