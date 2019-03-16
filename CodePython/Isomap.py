# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:32:47 2019

@author: Malo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

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



### ISOMAP ###

labels = donnees.iloc[:, 0:7] # labels = variables qualitatives

"""
 On va ici utiliser la méthode Isomap, en important le module Isomap de la classe manifold
 de la bibliothèque sklearn.

 Le but est de se ramener à un espace de 2 ou 3 dimensions tout en préservant les distances géodésiques
 (plus courte distance entre deux points sur une surface quelconque) entre les points.
 Pour cela, on va choisir k = nombre de voisins et calculer la matrice des distances géodésiques entre chaque 
 point et ses k voisins. 
 L'algorithme effectue ensuite une décomposition en éléments propres de la matrice. La réduction de dimension 
 se fait à partir des valeurs propres significatives.

 Questions : - comment choisir k ?
             - comment se passe la dernière étape ? (à partir des valeurs propres)



 La fonction utilisée par python est la suivante :
sklearn.manifold.Isomap(n_neighbors=5, n_components=2, 
                        eigen_solver=’auto’, tol=0, max_iter=None, 
                        path_method=’auto’, neighbors_algorithm=’auto’, n_jobs=None)

 EXPLICATION DES PARAMETRES
 n_neighbors = nombre de voisins à considérer pour chaque point
 n_components = nombre de dimensions auquel on veut réduire notre espace de variables
 eigen_solver = méthode de décomposition en éléments propres souhaitée (par défaut = 'auto', 
                utilise la méthode la plus adaptée) 
 tol et max_iter : dépendent du choix de eigen_solver si == 'dense'
 path_method = méthode utilisée pour trouver le plus court chemin/distance géodésique, par exemple Dijkstra
               (par défaut = 'auto', utilise la méthode la plus adaptée)
 neighbors_algorithm = algorithme utilisée pour la recherche des plus proches voisins
                       (par défaut = 'auto', utilise la méthode la plus adaptée)
 n_jobs = entier utilisé pour indiquer quelle quantité de mémoire on souhaite utiliser pour faire tourner l'algorithme
"""


features = donnees.iloc[:, 7:23].values # features = variables quantitatives
# Utilisation de la fonction Isomap expliquée plus haut. On choisit de réduire à 2 dimensions.
isomap = Isomap(n_neighbors = 10, n_components = 2) 
# On applique isomap sur nos données à l'aide de la méthode fit_transform, qui prend en paramètre nos données
# (ici features) et qui renvoie une matrice avec le même nombre de lignes mais avec un nombre de colonnes
# réduit à la dimension souhaitée.
features_transformed = isomap.fit_transform(features)
# On affiche sur un graphe nos résultats
plt.scatter(features_transformed[:, 0], features_transformed[:, 1], c = 'navy')
plt.show()

"""
 Questions : - à quoi servent les attributs embedding_, kernel_pca, etc
             - interpréter les différences de graphes quand on change n_neighbors et n_components
"""






















