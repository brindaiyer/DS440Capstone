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


#reading in and pre-processing data
columns = ['txId', 'timestep']
class_names = ['Illicit', 'Legal']

# Name the columns without known names
for x in range(165) :
    columns.append('col'+ str(x))


classes_csv = pd.read_csv('C:/Users/brind/Documents/archive/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
edgelist_csv = pd.read_csv('C:/Users/brind/Documents/archive/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
features_csv = pd.read_csv('C:/Users/brind/Documents/archive/elliptic_bitcoin_dataset/elliptic_txs_features.csv',names=columns)

#flattening data, appending class to features
data = features_csv.assign(result=classes_csv['class'])
data.to_csv(r'C:\Users\brind\Documents\startingdata.csv', index = None, header=True)
