# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:25:39 2019

@author: Bellec
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import numpy as np

donnees=pd.ExcelFile("/users/mmath/bellec/Téléchargements/Synthèse données des semis.xlsx").parse(1)
donnees=donnees.drop(index=[960,952])
donnees=donnees.drop('5°C T50 (h)',axis=1).drop('5°C T50 (j)',axis=1).drop('5°C TMG (h)',axis=1)

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
donnees.set_index('rep')
quanti=['5°C TMG (j)', 'Aire sous la courbe', '15 j', '16 j', '17 j','18 j', '19 j ', '20 j', '21 j', 'v15j-16j', 'v16j-17j', 'v17j-18j','v18j-19j', 'v19j-20j', 'v20j-21j']
quali=['Bancs', 'Pop', 'N° rep', 'camera', 'semis','zone', 'Echantillon', 'rep']

x=donnees.loc[:,quanti].values
y=donnees.loc[:,quali].values



from sklearn.manifold import TSNE 
for j in [5,30,60,100]:
    okok=donnees.loc[:,'semis'].drop_duplicates().values
    okok=okok[~pd.isna(okok)]
    target_ids = range(len(okok))
    colors = cm.rainbow(np.linspace(0, 1, len(okok)))
    y=donnees.loc[:,'semis'].values
    tsne = TSNE(n_components=2, n_iter=1000,perplexity=j)
    X_2d = tsne.fit_transform(donnees.loc[:,quanti].values)
    i=0
    #colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for label in okok:
        plt.scatter(X_2d[y == label, 0], X_2d[y == label, 1], c=colors[i], label=label)
        i=i+1
    plt.legend()
    plt.show()