# This is python code for preprocessing data.
#!/usr/bin/env python
# coding: utf-8

# # Méthodes d'imputation pour traiter des données manquantes, données aberrantes
# Données entrées: "Data_20201011_20210131_exemple.txt"
# 
# Résultat sortie : "Data_Imputed_example.txt"
# 
# Dans le fichier, nous avons des méthodes d'imputation:
# 
# - 1_Enlevement des données manquantes
# 
# def Enlevement(Data2_traite):
# 
# - 2_Remplacement_par_median
# 
# def Imputation_median(Data2_traite):
# 
# - 3_Remplacement_par_KNN (K plus proches voisins)
# 
# def Imputation_KNN(Data2_traite,nombre_voisin):
# 
# - 4_Imputation_par_Random_Forest
# 
# def Imputation_RF(Data2_traite):
# 
# - 5_Remplacement_par_l’estimation_à_noyau_(Kernel)
# 
# def Imputation_kernel(Data2_traite):
# 
# - 6_Remplacement_par_l’algorithme « expectation maximisation » de modèle de mélanges gaussiens (EM).
# 
# def Imputation_EM(Data2_traite):
# 
# - 7_Imputation par la regression stochatisque
# 
# def Imputation_regression(df_median):
# 
# 

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
from scipy import sparse
from random import choices


# In[2]:


#Indentifier des données aberrantes
def Traite_manquante(Data2_traite):
    #Des données manquantes
    df_median=Data2_traite.copy()
    feature2=df_median.columns
    ###Des données manquante#########
    #Des cellules vides
    #Traiter la température de zone 1 en avance#
    #Chercher des erreurs informatiques#
    for i in feature2:
        df_median[i] .loc[df_median[i]> 40000] = np.nan
    #Chercher des périodes où NB de fibrateur=0 sans arrêt de ligne#
    #Température ne peuvent pas égale à zéro#
    Temperature=['T° OVEN Zone 1','T° OVEN Zone 2', 'T° OVEN Zone 3', 'T° OVEN Zone 4']
    for i in Temperature:
        df_median.loc[df_median[i]==0,i] = np.nan #Température ne peuvent pas égale=0
    Fibrateur=['Nb de Fibrateurs en production']
    for i in Fibrateur:
        df_median.loc[(df_median['Vitesse de ligne (Oven)']>30) & (df_median['Nb de Fibrateurs en production'] ==0),i] = np.nan 
    for i in Fibrateur:
        df_median.loc[(df_median['Poids net produits']>0.5) & (df_median['Nb de Fibrateurs en production'] ==0),i] = np.nan  
    featureHung=feature2.drop(['Nb de Fibrateurs en production','Vitesse de ligne (Oven)'])
    for i in featureHung:
        df_median.loc[(df_median[i] ==0) & (df_median['Nb de Fibrateurs en production'] >3)&(df_median['Vitesse de ligne (Oven)']>0.5),i] = np.nan
    #Nb de fibrateur >3:
    for i in featureHung:
        df_median.loc[(df_median[i] ==0) & (df_median['Nb de Fibrateurs en production'] >3),i] = np.nan
    #Vitesse de ligne >0
    for i in featureHung:
        df_median.loc[(df_median[i] ==0) & (df_median['Vitesse de ligne (Oven)']>0.5),i] = np.nan
    for i in featureHung:
        df_median.loc[(df_median[i] ==0) & (df_median['Poids net produits']>0.5),i] = np.nan 
    return df_median


# In[13]:


#Les méthodes de traitement des données manquantes##
#1_Enlevement des données manquantes#
def Enlevement(Data2_traite):
    df_median=Data2_traite.dropna()
    return df_median

#2_Remplacement_par_median#
def Imputation_median(Data2_traite):
    df_median=Data2_traite.copy()
    feature2=Data2_traite.columns
    for i in feature2:
        df=df_median[i]
        df=df.dropna() #Enlever des cellules vides
        Median=df.median()
        df_median[i].loc[df_median[i].isnull()] = Median
    return df_median

#3_Remplacement_par_KNN (K plus proches voisins)#
from sklearn.impute import KNNImputer 
def Imputation_KNN(Data2_traite,nombre_voisin):
    #Imputation par k plus proches voisines#
    imputer = KNNImputer(n_neighbors=nombre_voisin)
    df_median=imputer.fit_transform(Data2_traite)
    df_median=pd.DataFrame(df_median, columns=Data2_traite.columns)
    return df_median
#4_Imputation_par_Random_Forest#

from missingpy import MissForest
def Imputation_RF(Data2_traite):
    df_median=Data2_traite.copy()
    imputer = MissForest(max_iter=10, n_estimators =100)
    df_median_tilde=[]
    df_median_tilde= imputer.fit_transform(df_median)
    df_median_tilde=pd.DataFrame(df_median_tilde,columns=df_median.columns)
    return df_median_tilde

#5_Remplacement_par_l’estimation_à_noyau_(Kernel)#
from sklearn.neighbors import KernelDensity
from distfit import distfit

def Imputation_kernel(Data2_traite):
    feature2=Data2_traite.columns
    df_median=Data2_traite.copy()
    #Remplacement par le kernel####
    for i in feature2:
        df_traite=df_median[i]
        df_traite = df_traite.dropna()
        #Kernel distribution#
        df_traite2=df_traite.to_numpy()
        dist = distfit()
        model=dist.fit_transform(df_traite2)
        #dist.plot() 
        #dist.plot_summary()
        print(i)
        dist1 = dist.model['distr']
        param=dist.model['params']
        for j in range(0, df_median.shape[0]):
            if np.isnan(df_median[i].loc[j]):
                df_median_tilde[i].loc[j] = dist1.rvs(*param[0:-2],loc=param[-2], scale=param[-1],size = 3)[0]
        t=[]
        param=[]
        dist1=[]
        dist=[]
        model=[]
    return df_median_tilde


#6_Remplacement_par_l’algorithme « expectation maximisation » de modèle de mélanges gaussiens (EM).
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
# from ..utils.sparsefuncs import _get_median
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.mixture import gaussian_mixture


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
# from ..utils.sparsefuncs import _get_median
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.mixture import gaussian_mixture

#EM Gaussian Mixtures algorithme pour l'imputation des données manquantes 
__all__ = [
    'MixtureImputer',
]


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask

class MixtureImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values="NaN", dist="gaussian",
                 axis=0, verbose=1, copy=True, min_complete=10,
                 n_components=2, covariance_type="full",
                 tol=0.001, max_iter=100):

        self.missing_values = missing_values
        self.dist = dist
        self.axis = axis
        self.verbose = verbose
        self.copy = copy
        self.min_complete = min_complete
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        Returns
        -------
        self : object
            Returns self.
        """
        if self.axis not in [0, 1]:
            raise ValueError("Can only impute missing values on axis 0 and 1, "
                             " got axis={0}".format(self.axis))

        # Since two different arrays can be provided in fit(X) and
        # transform(X), the imputation data will be computed in transform()
        # when the imputation is done per sample (i.e., when axis=1).
        if self.axis == 0:
            X = check_array(X, accept_sparse='csc', dtype=np.float64,
                            force_all_finite=False)

            if sparse.issparse(X):
                raise ValueError("Sparse matrices not supported yet")
                # self.statistics_ = self._sparse_fit(X,
                #                                     self.dist,
                #                                     self.missing_values,
                #                                     self.axis)
            else:
                dist=self.dist
                missing_values=self.missing_values
                axis=self.axis
                print(dist, missing_values,axis)
                self.statistics_ = self._dense_fit(X,
                                                   dist,
                                                   missing_values,
                                                   axis)

        return self

    def _dense_fit(self, X,dist, missing_values,axis ):
        """Fit the transformer on dense data."""
        X = check_array(X, force_all_finite=False)
        if(self.copy):
            X = np.copy(X)
        mask = _get_mask(X, missing_values)
        # masked_X = ma.masked_array(X, mask=mask)
        other_axis = int(not axis)
        is_missing_sample = (mask.sum(axis=other_axis) > 0)
        n_complete_samples = np.sum(~is_missing_sample)

        if n_complete_samples < self.min_complete:
            raise ValueError("Must have {0} complete samples, "
                             " but total complete samples={1}".
                             format(self.min_complete, n_complete_samples))

        imputer = {"gaussian": gaussian_mixture.GaussianMixture
                   }[dist](n_components=self.n_components,
                                covariance_type=self.covariance_type,
                                tol=self.tol, max_iter=self.max_iter)

        # Initial fit with complete cases
        imputer.fit(X[~is_missing_sample, :])

        return imputer

    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The input data to complete.
        """
        if self.axis == 0:
            check_is_fitted(self, 'statistics_')
            X = check_array(X, accept_sparse='csc', dtype=FLOAT_DTYPES,
                            force_all_finite=False, copy=self.copy)
            statistics = self.statistics_
            # if X.shape[1] != statistics.shape[0]:
            #     raise ValueError("X has %d features per sample, expected %d"
            #                      % (X.shape[1], self.statistics_.shape[0]))

        # Since two different arrays can be provided in fit(X) and
        # transform(X), the imputation data need to be recomputed
        # when the imputation is done per sample
        else:
            X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES,
                            force_all_finite=False, copy=self.copy)

            if sparse.issparse(X):
                raise ValueError("Sparse matrices not supported yet")

            else:
                statistics = self._dense_fit(X,
                                             self.strategy,
                                             self.missing_values,
                                             self.axis)

        # Get replancement index and impute initial prediction
        mask = _get_mask(X, self.missing_values)
        replace_indices = np.where(mask)[not self.axis]
        #Imputer des données
        for i in range (mask.shape[0]):
            replace_indices = np.where(mask[i,:])#La position de missing data de ligne i#
            random1=statistics.sample(1)#generer des valeurs aléatoire#
            random2=np.asarray(random1[0][0])
            for j in range(len(replace_indices)):
                X[i,replace_indices[j]] = random2[replace_indices[j]]
            random1=[]
            random2=[]
      
        # Fit then impute (repeat until convergence)
        for _ in range(self.max_iter):
            loglik_old = statistics.score(X)
            statistics.fit(X)                   
            # Impute
            # replace_indices = np.where(mask)[not self.axis]
            #X[mask] = np.take(prediction, replace_indices)
            #Imputer des données
            for i in range (mask.shape[0]):
                replace_indices = np.where(mask[i,:])#La position de missing data de ligne i#
                random1=statistics.sample(1)#generer des valeurs aléatoire#
                random2=np.asarray(random1[0][0])
                for j in range(len(replace_indices)):
                    X[i,replace_indices[j]] = random2[replace_indices[j]]
                random1=[]
                random2=[]
    
            loglik_new = statistics.score(X)
            if (loglik_new - loglik_old) < self.tol:
                break

        return X 
    def fit_transform(self, X, y=None, **fit_params):
        """Fit Gaussian mixture and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X, **fit_params).transform(X)


def Imputation_EM(Data2_traite):
    df_median_tilde=Data2_traite.copy()
    #Remplacer des valeurs extrêmes par la minimale et la maximale
    Max=df_median_tilde.max()
    Min=df_median_tilde.min()
    ###############################
    #Remplacement par le EM gaussian mixtures algorithme:
    imputer = MixtureImputer()
    X=df_median_tilde.copy()
    X.to_numpy()
    df_median_tilde_imputed = imputer.fit_transform(X)
    df_median_tilde_imputed=pd.DataFrame(df_median_tilde_imputed, columns=Data2_traite.columns)
    for i in df_median_tilde.columns:
        df_median_tilde_imputed[i] .loc[df_median_tilde_imputed[i]> Max[i]] = Max[i]
        df_median_tilde_imputed[i] .loc[df_median_tilde_imputed[i]< Min[i]] = Min[i]
    return df_median_tilde_imputed

#Imputation par la regression stochatisque
def Imputation_regression(df_median):
    feature=df_median.columns
    df_median[feature].isnull().sum()
    #Contruction des listes de missing paramètres#
    Parameter=[] #La liste des paramètres complètes#
    for i in feature:
        if(df_median[i].isnull().sum()==0):
            Parameter.append(i)

    missing_columns=list(set(feature) - set(Parameter))

    def random_imputation(df, feature):
        number_missing = df[feature].isnull().sum()
        observed_values = df.loc[df[feature].notnull(), feature]
        df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
        return df
    #Enlèvement des données#
    df_median2=df_median.copy()
    df_median2=df_median2.dropna(axis=0)

    random_data = pd.DataFrame(columns = [missing_columns])
    from sklearn import linear_model
    feature=df_median2.columns
    for j in feature:
        df_median[j+ '_imp'] = df_median[j]      
    for i in missing_columns:     
        random_data[i] = df_median[i]   
        model = linear_model.LinearRegression()
        model.fit(X = df_median2[feature], y = df_median2[i])
        #Remplir des données randomly
        df_median3=df_median.copy() 
        for j in feature:
            df_median3[j+ '_imp'] = df_median3[j]
            df_median3 = random_imputation(df_median3,j)
        predict = model.predict(df_median3[feature+'_imp'])
        std_error = (predict[df_median[i].notnull()] - df_median.loc[df_median[i].notnull(), i + '_imp']).std()
        random_data[i]=predict
        #observe that I preserve the index of the missing data from the original dataframe
        random_predict = np.random.normal(size = df_median[i].shape[0], loc = predict, scale = std_error)
        random_data.loc[(df_median[i].isnull()) & (random_predict > 0),  i] = random_predict[(df_median[i].isnull()) & (random_predict > 0)]
                                                                         
    for i in missing_columns:
        df_median=df_median.drop([i+ '_imp'],axis=1)

    for i in missing_columns:
        df_median[i]=random_data[i]
    return df_median


# In[5]:


#Ensemble de données de 11/10/2020 - 31/12/2020
filename = os.path.join(os.getcwd(), "Data_20201011_20210131_exemple.txt")
df = pd.read_csv(filename, sep="\t")
df


# In[6]:


#Enlevement de idSKU, ProductName, ProductForm, TimeMeasure
df2=df.drop(['idSKU', 'ProductName', 'ProductForm', 'TimeMeasure'], axis=1)


# In[7]:


#Enlevement des données aberrantes
df2=Traite_manquante(df2)


# In[8]:


#La quantité des données manquantes + données aberrantes 
df2.isna().sum(axis=0)


# In[9]:


#Imputation par different méthode (Retirer '#' pour l'utiliser)
#Enlevement des données manquantes
#df_Imputed1=Enlevement(df2)
#Imputation par le median
#df_Imputed2=Imputation_median(df2)
#Imputation par KNN
#df_Imputed3=Imputation_KNN(df2,nombre_voisin=2)
#Imputation par RF
#df_Imputed4=Imputation_RF(df2)
#Imputation par l'estimation à noyau 
#df_Imputed5=Imputation_kernel(df2)
#Imputation par Expectation et Maximisation
#df_Imputed6=Imputation_EM(df2)
#Imputation par la regression stochastique
df_Imputed7=Imputation_regression(df2)


# In[10]:


#Calculer la vitesse de ligne théorique et le grammage
def Calcule_vitess_grammage(df_median):
    Grammage=df_median['Densité nominal']*df_median['Epaisseur']/1000
    df_median['Grammage']=Grammage
    Vitess=df_median['Nb de Fibrateurs en production']*600*(1+df_median['LOI nominal']/100)/(1.3*df_median['Grammage']*60)
    df_median['Vitesse de ligne theorie']=Vitess
    return df_median


# In[11]:


df_Imputed7=Calcule_vitess_grammage(df_Imputed7)


# In[14]:


#Sauvegarde le fichier des données en .csv
df_Imputed7.to_csv('Data_Imputed_example.txt',sep='\t')


# In[ ]:



