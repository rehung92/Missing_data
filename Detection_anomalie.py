#!/usr/bin/env python
# coding: utf-8

# 
# # La détection d'anomalie par le fôrest d'isolation et la classification
# Données entrées: "Data_Imputed_example.txt", "History_202102_202108.txt"
#                     Classifieur :"KNN20220209.joblib"
#                     
# Résultat sortie : "Classification_et_anomalie.txt" (Label = -1 signifie "anomalie")
# 
# La fonction Detection d'anomalie sur les données
# def Anomalie(df,Historique, Length, Classifier2):
# a 4 entrées : 
# 
# - df : le dataframe des données à traiter
# - Historique : le dataframe des historique
# - Length : traiter des observations de "df" depuis le début jusqu'à observations à la position Length, normalement Length prend toutes les observations de "df" (Legnth=len(df));
# - Classifier2 : le classifieur qu'on utilise pour classer des observations normales (par exemple: classifieur de KNN, random forest ou SVM)
# 
# Le résultat soit la classe de fonctionnement pour des observations normales (numéroté de 1 à 9 selon des états de fonctionnement) soit -1 signifiant une anomalie.

# In[14]:


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
import random
print('seaborn '+ random.__version__)
from sklearn import preprocessing
#Scaler pour normaliser des données
from sklearn.preprocessing import LabelEncoder



# In[2]:


#Creation de isolation forest#
def classify_data(df): 
    if len(df)!=0:
        data=df.values   
        label_column = data[:, -1]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        classification = unique_classes[index] 
    else:
        classification=-1
    return classification
#Selection de feature des données sauf 'Nombre de fibrateur' or 'feature1' car nous distinguons l'arret et les autres états:
def select_feature(data): 
    T=data.columns
    return random.choice(T)
#Selection de valeur à diviser :
def select_value(data,feat):
    mini = data[feat].min()
    maxi = data[feat].max()
    return (maxi-mini)*np.random.random()+mini
#Diviser des données :
def split_data(data, split_column, split_value):    
    data_below = data[data[split_column] <= split_value]
    data_above = data[data[split_column] >  split_value]
    return data_below, data_above
#Construction de isolationTree :
Temperature=['feat11','feat17'] #Chercher l'incendie à la hotte entrée feat11=zone0 et à la hotte sortie feat17=zone6#
def isolation_tree(data,feat1,Temperature,counter=0, max_depth=50,random_subspace=False):
    Temperature1=Temperature.copy()
    # End Loop if max depth or isolated
    if (counter == max_depth) or data.shape[0]<=1 :
        classification = classify_data(data)
        return classification
    else:
        # Counter
        counter +=1
        if counter==1:
          split_column=feat1
          split_value = 4
        # Select feature
        elif counter==2:
          Hung=random.choice(Temperature1)
          split_column=Hung
          split_value = 300
          Temperature1.remove(Hung)
        elif counter==3:
          Hung2=random.choice(Temperature1)
          split_column=Hung2
          split_value = 300
        else:
          split_column = select_feature(data)       
        # Select value
          split_value = select_value(data,split_column)  
        # Split data
        data_below, data_above = split_data(data,split_column,split_value)    
        # instantiate sub-tree      
        question = "{} <= {}".format(split_column, split_value)
        sub_tree = {question: []}      
        # Recursive part
        below_answer = isolation_tree(data_below,feat1,Temperature, counter,max_depth=max_depth)
        above_answer = isolation_tree(data_above,feat1,Temperature, counter,max_depth=max_depth)       
        if below_answer == above_answer:
            sub_tree = below_answer
        else:
            sub_tree[question].append(below_answer)
            sub_tree[question].append(above_answer)      
        return sub_tree
#Construction de isolatuion forest:
def isolation_forest(df,feat1,Temperature,n_trees=50, max_depth=15, subspace=256):
    forest = []
    for i in range(n_trees):
        # Sample the subspace
        if subspace<=1:
            df = df.sample(frac=subspace)
        else:
            df = df.sample(subspace)        # Fit tree
        tree = isolation_tree(df,feat1,Temperature,max_depth=max_depth)    
        # Save tree to forest
        forest.append(tree)    
    return forest
#La longeur de path :
def pathLength(example,iTree,path=0,trace=False):    
    # Initialize question and counter
    path=path+1
    question = list(iTree.keys())[0]
    feature_name, comparison_operator, value = question.split()   
    # ask question
    if example[feature_name].values <= float(value):
        answer = iTree[question][0]
    else:
        answer = iTree[question][1]          
    # base case
    if not isinstance(answer, dict):
        return path    
    # recursive part
    else:
        residual_tree = answer
        return pathLength(example, residual_tree,path=path)
    return path
# Evaluer une instance:
def evaluate_instance(instance,forest):
    paths = []
    for tree in forest:
        paths.append(pathLength(instance,tree))
    return paths
#Calculate the pathlength moyenne de isolation forest#
def Path_average(DataFrame,IForest):
    Path_Hung=[]
    Path_moyenne=0 #Initialize the pathlength moyenne#
    for i in range(len(DataFrame)):
        T=np.array(DataFrame.iloc[i])
        T=T.reshape(1,DataFrame.shape[1])
        T=pd.DataFrame(T, columns=Columns_name)
        path_len=evaluate_instance(T,IForest)
        path_len=np.mean(path_len)
        Path_Hung.append(path_len)
        Path_moyenne+=path_len
    Path_moyenne=Path_moyenne/len(DataFrame)
    return Path_moyenne, Path_Hung


# In[3]:


#Detection d'anomalie sure les données
def Anomalie(df,Historique, Length, Classifier2):
    Label_nouvelle=[]
    scaler=preprocessing.MinMaxScaler()
    scaler.fit(Historique.drop(['Label'],axis=1))
    df2=scaler.transform(df)
    #Normaliser les labels
    Label_encode=LabelEncoder()
    #History_brut.columns=Columns_name
    History_brut=Historique.copy()
    History_brut['Label']=Label_encode.fit_transform(Historique['Label'])
    #Create different classe dataframe pour toutes les classes#
    Num_classe=History_brut['Label'].max()+1 #Nombre de classe dans l'historique
    for i in range(0,int(Num_classe)):
        globals()['df_classe'+str(i)]=History_brut.loc[History_brut['Label'] == i]
        globals()['df_classe'+str(i)]=globals()['df_classe'+str(i)].drop(['Label'], axis=1)
        globals()['df_classe'+str(i)].columns=Columns_name

    for j in range(0,int(Num_classe)):
            globals()['T'+str(j)]=globals()['df_classe'+str(j)].sample(n=300,replace=True)

    Nouvelle_donnees=pd.DataFrame(np.array(df),columns=Columns_name) 
    Nouvelle_donnees2=pd.DataFrame(df2,columns=Columns_name) #Données normalisées pour la classification
    i=0
    while i < Length:
        print (i)
        Anomalie=Nouvelle_donnees.iloc[i]
        Anomalie=np.array(Anomalie)
        Anomalie=Anomalie.reshape(1,Nouvelle_donnees.shape[1])
        Anomalie=pd.DataFrame(Anomalie, columns=Columns_name)
        #Teste pour chaque classe:
        Note_anomalie=0 #=0 si l'instance ne se trouve pas dans toutes les classes.
        for j in range(0,int(Num_classe)):
            H2=[]
            H2=globals()['T'+str(j)].copy()
            H2.append(Anomalie)
            H3=H2.copy()
            IForest10=isolation_forest(H3,feat1,Temperature,n_trees=15, max_depth=20) #Creation de isolationForest
            path_len3=evaluate_instance(Anomalie,IForest10)
            #Path_moyenne, Path_Hung=Path_average(globals()['T'+str(j)],IForest10)
            Path_moyenne, Path_Hung=Path_average(H3,IForest10)
            print(np.mean(path_len3), Path_moyenne, 0.65*Path_moyenne)
            if (np.mean(path_len3)<0.65*Path_moyenne): #Le path est plus court que la moyenne des références-> anomalie pour cette classe.
                Note_anomalie+=0
            else:
                Note_anomalie+=1
        if Note_anomalie==0:
            Label_nouvelle.append(-1) #Anomalie de toutes classes actuelles -> à classifier après
        else:
            Anomalie=np.array(Nouvelle_donnees2.iloc[i])
            Anomalie=Anomalie.reshape(1,Nouvelle_donnees2.shape[1])
            G=(Classifier2.predict(Anomalie))[0]
            Label_nouvelle.append(G)
            print(G)
        Position.append(i)
        i+=1
    return Label_nouvelle


# In[4]:


#Chargement des données
#Chargement de l'historique
filename1 = os.path.join(os.getcwd(), "History_202102_202108.txt")
Historique = pd.read_csv(filename1, sep='\t')
Historique=Historique.drop(['Unnamed: 0'], axis=1)
#Chargement de nouvelles données
filename = os.path.join(os.getcwd(), "Data_Imputed_example.txt")
df = pd.read_csv(filename, sep='\t')
t=Historique.columns.tolist()
t.remove('Label')
df=df[t] #Prendre des paramètres similaires à Historique


# In[5]:


Historique.columns


# In[6]:


df.columns


# In[7]:


df


# In[8]:


#Charger le classifier développer dans l'étape 3_Classification pour classer des données:
import joblib
filename4=os.path.join(os.getcwd(),"KNN20220209.joblib") #Classifier de types K plus proches voisins
Classifier2 = joblib.load(filename4)


# In[9]:


#Construction de isolationForest#
Label_nouvelle=[] #Label de chaque instance de 20210201_20210630 =-1 si anomalie sinon classifier Randomforest.
Position=[]
feat1='feat1' #Nombre de fibrateurs
Temperature=['feat11','feat17'] #Température de hotte entrée et hotte sortie
Length=len(df)
Columns_name=['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8'
,'feat9','feat10','feat11','feat12','feat13','feat14','feat15','feat16','feat17','feat18'
,'feat19','feat20','feat21','feat22','feat23','feat24','feat25','feat26','feat27','feat28'
,'feat29']


# In[10]:


Label_nouvelle=Anomalie(df, Historique, 10, Classifier2)


# In[11]:


Label_nouvelle=pd.DataFrame(Label_nouvelle, columns=['Label'])
Label_nouvelle.to_csv('Classification_et_anomalie.txt',sep='\t')


# In[ ]:



