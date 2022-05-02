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

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import seaborn as sns
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn import preprocessing 
from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve,roc_auc_score,fbeta_score,recall_score
from sklearn.metrics import plot_precision_recall_curve,average_precision_score,f1_score
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn.svm import SVC

#data - preprocessing all done
id_time=["txId", "time_step"]
feature_names = ['feature_'+str(i) for i in range(1,166)]
column_names = id_time + feature_names
orig2contiguos_csv = pd.read_csv('C:/Users/brind/Documents/DS440/EvolveGCN/data/elliptic_bitcoin_dataset/elliptic_txs_orig2contiguos.csv')
classes_csv = pd.read_csv('C:/Users/brind/Documents/DS440/archive/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
#classes_csv.columns = ['txId', 'class_label']
edgelist_csv = pd.read_csv('C:/Users/brind/Documents/DS440/archive/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
nodetime_csv = pd.read_csv('C:/Users/brind/Documents/DS440/EvolveGCN/data/elliptic_bitcoin_dataset/elliptic_txs_nodetime.csv')
edgelisttimed_csv = pd.read_csv('C:/Users/brind/Documents/DS440/EvolveGCN/data/elliptic_bitcoin_dataset/elliptic_txs_edgelist_timed.csv')
features_csv = pd.read_csv('C:/Users/brind/Documents/DS440/archive/elliptic_bitcoin_dataset/elliptic_txs_features.csv',names=column_names)

##
classes_csv['class'].value_counts() #157205 rows total
##
features_csv.columns = ['id', 'time'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
features_csv.head()
##
features_csv['time'].value_counts().sort_index().plot();
plt.title('Number of transactions in each time step');
##
### merge with classes
features_csv = pd.merge(features_csv, classes_csv, left_on='id', right_on='txId', how='left')
plt.figure(figsize=(12, 8))
grouped = features_csv.groupby(['time', 'class'])['id'].count().reset_index().rename(columns={'id': 'count'})
plt.title('Number of transactions in each time step');
sns.lineplot(x='time', y='count', hue='class', data=grouped);
plt.legend(loc=(1.0, 0.8));
plt.title('Number of transactions in each time step by class');
###plt.show()
##
####
features_csv.head()
features_csv=features_csv.rename(columns={"class":"Class"})
cleaned_df = features_csv.copy()
cleaned_df.pop('time')
#cleaned_df.pop('txId')
cleaned_df.pop('id')
####
print(cleaned_df)

#cleaned_df = cleaned_df.fillna(-1)
for i in range(len(cleaned_df)):
    if cleaned_df.iloc[i,165] == "unknown":
        cleaned_df.iloc[i,165] = -1
    elif cleaned_df.iloc[i,165] == 1:
        cleaned_df.iloc[i,165] = 1
    elif cleaned_df.iloc[i,165] == 2:
        cleaned_df.iloc[i,165] = 0

##
###Create array of unknown Class randomly with the same proportion 1/10 between "zeros" and "ones"
def rand_bin_array(K, N):
    arr = np.zeros(N,int)

    arr[:K]  = int( 1)
    np.random.shuffle(arr)
    return arr

### Put together all Classes
prueba_0=cleaned_df[cleaned_df['Class']==2] 
prueba_0
####
prueba_1=cleaned_df[cleaned_df['Class']==1] 
prueba_1
####
prueba=cleaned_df[cleaned_df['Class']=="unknown"]  
prueba
##

prueba['Class']=rand_bin_array(15720,157205)   # 3) Change Class array -1  target with a relation 1/10  Illicit-Licit
vertical = pd.concat([prueba, prueba_0,prueba_1], axis=0)
vertical=vertical.sort_index()   # 5) Order again 
vertical 

##
y=vertical['Class']
data= vertical.copy()
##
###the distribution of target variable - Class
vc = data['Class'].value_counts().to_frame().reset_index()
vc['percent'] = vc["Class"].apply(lambda x : round(100*float(x) / len(data), 2))
vc = vc.rename(columns = {"index" : "Target", "Class" : "Count"})
vc #only 8% of transactions are considered to be fraud
##
###taking a sample of only 1000 fraud cases
### We consider Fraud Like "1" and Not Fraud like "0"
non_fraud = data[data['Class'] == 0].sample(1000)
fraud = data[data['Class'] == 1]
##
df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
X = df.drop(['Class'], axis = 1).values
X = X.astype('float')

Y = df["Class"].values
##
print(X.shape)

###visualizing fraud and non-fraud transactions
def tsne_plot(x1, y1, name="graph.png"):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth=1, alpha=0.8, label='Fraud')
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth=1, alpha=0.8, label='Non Fraud')

    plt.legend(loc='best');
    plt.savefig(name);
    #plt.show();
    
tsne_plot(X, Y, "original.png")
##
###Autoencoders - helps to model the identity function
###input layer
input_layer = Input(shape=(X.shape[1],))
#### encoding part
encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(50, activation='relu')(encoded)
#### decoding part
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)
#### output layer
output_layer = Dense(X.shape[1], activation='relu')(decoded)
##
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")
##
###min max scaling on data
x = data.drop(["Class"], axis=1)
y = data["Class"].values

##
x_scale = preprocessing.MinMaxScaler().fit_transform(x.values)
x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]

##
autoencoder.fit(x_norm[0:2000], x_norm[0:2000], 
                batch_size = 256, epochs = 10, 
                shuffle = True, validation_split = 0.20); #validation part

###obtain the latent representations
hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])

##
###hidden representations of fraud and non-fraud
norm_hid_rep = hidden_representation.predict(x_norm[:4000])
fraud_hid_rep = hidden_representation.predict(x_fraud)

##
###visualize latent representations: fraud vs non-fraud
rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)
y_n = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(fraud_hid_rep.shape[0])
rep_y = np.append(y_n, y_f)


##tsne_plot(rep_x, rep_y, "latent_representation.png")
##
###simple linear classifier and metrics
train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)

clf = LogisticRegression(solver="lbfgs",max_iter=4000).fit(train_x, train_y)
pred_y = clf.predict(val_x)
print(pred_y)

##
print ("")
print ("Classification Report: ")
print (classification_report(val_y, pred_y))
##
print ("")
print ("Accuracy Score: ", accuracy_score(val_y, pred_y))
##
###Model Evaluation using Confusion Matrix
cnf_matrix = metrics.confusion_matrix(val_y, pred_y)
cnf_matrix
##
total1=sum(sum(cnf_matrix))
##
###calculate accuracy
#######from confusion matrix calculate accuracy
accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : {0:0.2f}'.format(accuracy1))
##
###calculate sensitivity
sensitivity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : {0:0.2f}'.format(sensitivity1 ))
##
###calculate specificity
specificity1 = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity : {0:0.2f}'.format(specificity1))
##
###Visualizing Confusion Matrix using Heatmap
###%matplotlib inline
##
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
### create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
##
### predict probabilities and create ROC Curve
yhat = clf.predict_proba(val_x)

# retrieve just the probabilities for the positive class
pos_probs = yhat[:, 1]

#probabilities for negative class
neg_probs = yhat[:,0]

# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(val_y, pos_probs)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
##
svc = SVC(random_state=42)
svc.fit(train_x, train_y)

svc=svc_disp = plot_roc_curve(svc, val_x, val_y)
#plt.show()



#ROC AUC
ROC_AUC=roc_auc_score(val_y, clf.predict_proba(val_x)[:, 1])
print('ROC_AUC: {0:0.2f}'.format(ROC_AUC))
##
#FB-Score
F1=fbeta_score(val_y,yhat[:, 1].round(),beta=1)
F2=fbeta_score(val_y,yhat[:, 1].round(),beta=2)
F_0_5=fbeta_score(val_y,yhat[:, 1].round(),beta=0.5)

print('F1-score: {0:0.2f}'.format(F1),'F2-score:  {0:0.2f}'.format(F2), 'F_0.5 score:  {0:0.2f}'.format(F_0_5 ))

#F1-Score
F1=f1_score(val_y,yhat[:, 1].round(), average=None)
print('F1_score: {0:0.2f}'.format(F1[1]))
##
#Accuracy Score
AS=accuracy_score(val_y,yhat[:, 1].round())
print('Accuracy Score: {0:0.2f}'.format(AS))

#Average Precision Recall Score
average_precision = average_precision_score(val_y,yhat[:, 1].round())
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

#precision recall curve
disp = plot_precision_recall_curve(clf, val_x, val_y)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))



