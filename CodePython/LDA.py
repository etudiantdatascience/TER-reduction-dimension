# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:00:50 2019

@author: Malo
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

### Import des données expertise/nettoyage ###

# Linux
# donnees = pd.ExcelFile("/users/gillard/Downloads/Synthese donnees des semis.xlsx").parse(1)
# Windows
donnees = pd.ExcelFile("C:/Users/Malo/Downloads/Synthese donnees des semis.xlsx").parse(1)
donnees=donnees.drop(index=[960,952])
print('Pourcentage de NaN: ','\n',round(donnees.isna().sum()/960*100,1))
donnees=donnees.drop('5°C T50 (h)',axis=1).drop('5°C T50 (j)',axis=1).drop('5°C TMG (h)',axis=1)

# On rajoute les vitesses de germination dans les variables quantitatives
liste=[]
col={}
j=0
for i in range(10,16):
    a=donnees.columns[i]
    b=donnees.columns[i+1]
    liste.append(donnees[b]-donnees[a])
    c='v'+a+'-'+b
    col[j]=c.replace(' ','')
    j=j+1
croissance=pd.DataFrame(liste).T
croissance=croissance.rename(col, axis='columns')
donnees=pd.concat([donnees,croissance],axis=1).dropna()
donnees = donnees.set_index('rep')

### LDA ###

"""
 On va ici utiliser la méthode LDA, en important les modules LinearDiscriminantAnalysis, train_test_split
 et StandardScaler de la bibliothèque sklearn.
 L'objectif est de trouver une combinaison linéaire des variables qui caractérisent ou séparent 2 ou plus
 classes d'individus.
"""

# Préparation des données
j = 1
labels = donnees.iloc[:, j].values.astype('str')  # labels = variables qualitatives
features = donnees.iloc[:, 7:23].values # features = variables quantitatives



"""
train_test_split :
La fonction train_test_split va séparer aléatoirement nos données(variables quantitatives et qualitatives)
en plusieurs set : un set de variables "entrainées" (qui forment plusieurs groupes prédéfinis selon leurs
caractéristiques/valeurs), un set de variables quantitatives "test" (sur lesquelles on appliquera la LDA,
c'est à dire que l'on va chercher à savoir à quel groupe prédéfini ces variables appartiennent, en se basant
sur les variables "entrainées"). On fait la même chose avec nos variables qualitatives.

On a donc la fonction suivante :
sklearn.model_selection.train_test_split(*arrays,test_size, train_size, random_state, shuffle, stratify)

EXPLICATION DES PARAMETRES
arrays = listes/matrices/dataframe d'index de même taille (on a en entrée un dataframe de variables quantitatives
         et un dataframe de variables qualitatives, pour un même nombre d'individus)
test_size = proportion du jeu de données qui sera testée
train_size = proportion du jeu de données qui sera entrainée (si = None, c'est la valeur complémentaire de test_size
             qui est prise)
random_state = paramètre qui gère le côté aléatoire du "split" : si égal à un entier, alors on aura toujours
               les mêmes données test/entrainées en sortie. Siégal None, ces sorties seront aléatoires
shuffle = booléen. Si égal à True alors les données seront mélangées avant d'être "split"
stratify = None si shuffle = True. Sinon, les données seront "split" par couche superposée (traduction approximative)

EN SORTIE
features_train =  données (variables quantitatives) entrainées
features_tets = données (variables quantitatives) test
labels_train = données (variables qualitatives) entrainées  
labels_test = données (variables qualitatives) test                                         

"""

features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels, test_size=0.2, random_state=0) 

# Feature scaling : on récupère des variables centrées-réduites
sc = StandardScaler() # méthode de sklearn.preprocessing pour centrer et réduire nos données quantitatives
features_train = sc.fit_transform(features_train) # On centre et réduit les données quantitatives entrainées
features_test = sc.transform(features_test) # On centre et on réduit les données quantitatives test

# Application de la LDA

"""
On utilise la méthode LDA de la classe sklearn.discriminant_analysis :
sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, 
                                                         n_components=None, store_covariance=False, tol=0.0001)
EXPLICATION DES PARAMETRES
solver = méthode utilisée pour la réduction de dimension, par défaut égal à svd (décomposition en valeur singulière)
shrinkage = paramètre de réduction en lien avec la paramètre solver, si celui-ci n'est pas 'svd'
priors = ???
n_components = nombre de dimensions auxquelles on souhaite réduire
store_covariance = ???
tol = seuil de tolérance pour la 'svd'

"""

lda = LDA(n_components = 2) # Utilisation de LDA pour réduire notre espace de variables à deux dimensions
features_train = lda.fit_transform(features_train, labels_train) # On applique LDA sur nos données quantitatives entrainées
features_test = lda.transform(features_test) # On applique LDA sur nos données quantitatives test


### EXPLICATIONS A VENIR

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(features_train, labels_train)
 
# Predicting the Test set results
y_pred = classifier.predict(features_test)

# Représentation graphique
plt.show()
for i in features_test: 
    plt.scatter(i[0],i[1])
plt.show()
for i in features_train: 
    plt.scatter(i[0],i[1])
plt.show()




