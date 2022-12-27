#!/usr/bin/env python
# coding: utf-8

# # Traiter le problème des classes déséquilibrées avec la méthode SMOTE
# 
# Données entrées:  "History_202102_202108.txt" 
# 
# Résultat sortie : "History_oversample.txt"

# In[2]:


import os
from numpy.lib.shape_base import column_stack
clear=lambda: os.system('clear')
clear()
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier #Classsification par les plus proches voisins
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC
#Sauvegarde des classifieurs
import pickle
import joblib
from joblib import dump
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


# In[2]:


#Historique de 11_10_2020-31_01_2021##
filename1 = os.path.join(os.getcwd(), "History_202102_202108.txt")
Historique = pd.read_csv(filename1, sep='\t')
Historique=Historique.drop(['Unnamed: 0'], axis=1)


# In[4]:


#Construction Historique de 2000 obersvations
#Create different classe dataframe pour toutes les classes#
Num_classe=Historique['Label'].max()+1 #Nombre de classe dans l'historique
for i in range(0,Historique['Label'].max()+1):
    globals()['df_classe'+str(i)]=Historique.loc[Historique['Label'] == i]

for j in range(1,Num_classe):
    if(len(globals()['df_classe'+str(j)])>1000):
        globals()['T'+str(j)]=globals()['df_classe'+str(j)].sample(1000)
    else:
        globals()['T'+str(j)]=globals()['df_classe'+str(j)].sample(int(len(globals()['df_classe'+str(j)])))

#Concatenate dataframe
Data=[globals()['T'+str(j)]for j in range(1,Num_classe)]
Data = pd.concat(Data)


# In[5]:


data=Data.values
X, y = data[:, :-1], data[:, -1]
Label_encode=LabelEncoder().fit(y)
y1 = Label_encode.fit_transform(y)
# transform the dataset
oversample = SMOTE()
X, y1 = oversample.fit_resample(X, y1)
y2=Label_encode.inverse_transform(y1)


# In[6]:


columns_name=Historique.columns
columns_name=columns_name.drop(['Label'])
Data_oversample=pd.DataFrame(X, columns=columns_name)
Data_oversample['Label']=y2
Data_oversample.to_csv('History_oversample.txt',sep='\t')
