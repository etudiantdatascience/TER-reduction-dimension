# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:38:42 2019

@author: Reunan
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ImportTER import FonctionImportDonnees
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import make_classification

#X, y = make_classification(n_samples=1000, n_features=4,
#                           n_informative=4, n_redundant=0,
#                           random_state=0, shuffle=False,n_classes=4,n_clusters_per_class=1)


donnees,quanti,quali=FonctionImportDonnees("/users/mmath/bellec/Téléchargements/Synthèse données des semis.xlsx")



for variableQuali in quali:
    valeur=donnees[variableQuali].drop_duplicates().values
    donnees[variableQuali]=donnees[variableQuali].replace(valeur,list(range(len(valeur))))

importance=np.zeros(len(quanti))

for i in quali:
    clf = RandomForestClassifier(n_estimators=50, max_depth=15,
                                 random_state=4,criterion='gini')
    clf.fit(donnees[quanti],donnees['zone'])
    importance=importance+clf.feature_importances_

feature_imp = pd.Series(importance,index=quanti).sort_values(ascending=False)
x=np.arange(len(feature_imp.values))
fig, axes = plt.subplots(figsize=(15,3))
plt.bar(x,feature_imp.values)
plt.xticks(x, feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
#print(clf.predict([[0, 0, 0, 0]]))

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(donnees[quanti],donnees['zone'],test_size=0.3,random_state=4)

forest=RandomForestClassifier(n_estimators=500, max_depth=15,criterion='gini')

forest = forest.fit(X_train,y_train)
print(1-forest.oob_score)
print(1-forest.score(X_test,y_test))


param_grid_rf = { 'n_estimators' : [100, 500],
                 'max_depth': [4,5,84]}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state= 5), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
print ("Score final : ", round(grid_search_rf.score(X_train, y_train) *100,4), " %")
print ("Meilleurs parametres: ", grid_search_rf.best_params_)
print ("Meilleure config: ", grid_search_rf.best_estimator_)