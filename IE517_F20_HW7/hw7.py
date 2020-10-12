#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:32:23 2020

@author: liumengqi
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

df = pd.read_csv('/Users/liumengqi/Downloads/ccdefault.csv')

X, y = df[df.columns[1:-1]].values, df['DEFAULT'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1,stratify=y)

#Part 1: Random forest estimators


param_grid = {'n_estimators':[10,20,50,100,125,150,200]}
rf = RandomForestClassifier(criterion='gini',random_state=1,n_jobs=2)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=10,n_jobs=-1,verbose=1,return_train_score=True)
grid_search
grid_search.fit(X_train, y_train)

mean_train_score =grid_search.cv_results_['mean_train_score']
print(mean_train_score)

mean_test_score =grid_search.cv_results_['mean_test_score']
print(mean_test_score)

grid_search.best_params_

#Part 2: Random forest feature importance

#build the randomforestclassifier model with the best parameters
rfbest = RandomForestClassifier(n_estimators=200,random_state=42)
rfbest.fit(X_train,y_train)

importances = rfbest.feature_importances_
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))
dX = df.iloc[:, 0:23]
df.feature_names = list(dX.columns.values) 
df.class_names = df.columns[23]
labels =np.array(df.feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.show()

print("Feature Importances:")
for i in range(X_train.shape[1]):
    print(labels[i], importances[sorted_index[i]])
    
    
print("My name is {Mengqi Liu}")
print("My NetID is: {mengqi3}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

