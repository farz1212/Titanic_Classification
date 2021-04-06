# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:27:04 2018

@author: Farzaad
"""
#Import statements
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier as rr
from sklearn.model_selection import train_test_split as tts
import numpy as np
from sklearn.metrics import average_precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.neighbors import KNeighborsClassifier as kn
import matplotlib.pyplot as plt
import seaborn as sns
lst = []

#Read file
df = pd.read_csv("titanic3.csv")

#Clean data
df.dropna(thresh=df.shape[0]*0.1,how="all",axis=1,inplace=True)
df["age"].fillna(df["age"].median(),inplace=True)
df["embarked"].fillna(df["embarked"].mode(),inplace=True)
df["family_size"] = df["parch"] + df["sibsp"] + 1
df["fare"].fillna(float(df["fare"].mode()),inplace=True)
df["home.dest"].fillna(method="ffill",inplace=True)

# Mapping
df["sex"] = df["sex"].map({"male":1,"female":0}).astype(int)
df["title"] = df["name"].map(lambda name : name.split(".")[0].split(" ")[1])
df["title"] = df["title"].map({"Mr" : "Mr", "Mrs" : "Mrs", "Miss" : "Miss", "Master" : "Master"})
df["title"].fillna("Others", inplace=True)
#print(df["title"])
df["title"] = df["title"].map({"Mr" : 0, "Mrs" : 1, "Miss" : 2, "Master" : 3, "Others" : 4})
df = pd.concat([df, pd.get_dummies(df["embarked"])],axis=1)

#Dropped columns
df.drop(["name", "ticket", "cabin","boat","sibsp","parch","embarked","home.dest"], axis=1, inplace=True)
#for x in range(len(list(df))):
#    if abs(df[list(df)[x]].corr(df["survived"])) < 0.03:
#        lst.append(list(df)[x])
#            
#df.drop(lst,axis=1,inplace=True)          

#Correlation
heat = sns.heatmap(df[list(df)].corr(),annot=True)
plt.show()

#No NaN
print(df.isna().sum())
print(df.head(5))

#Split 
train = df.drop("survived",axis=1)
test = df["survived"]
X_train, X_test, y_train, y_test = tts(train,test,test_size=0.3)

#Classifiers
op_dict={}
pre_dict={}
rec_dict={}

algos = [SVC(), rr(n_estimators = 30), kn(), dt()]

for clf in algos:
    clf.fit(X_train,y_train)
    op_dict[str(clf)[:3]] = clf.score(X_test,y_test)
    average_precision = average_precision_score(y_test, clf.predict(X_test))
    pre_dict[str(clf)[:3]] = average_precision
    rcall = recall_score(y_test, clf.predict(X_test))
    rec_dict[str(clf)[:3]] = rcall
    

#clf = SVC()
#clf.fit(X_train,y_train)
#op_dict['SVM'] = clf.score(X_test,y_test)
#average_precision = average_precision_score(y_test, clf.predict(X_test))
#pre_dict["SVM_precision"] = average_precision
#
#clf=rr(n_estimators=40)
#clf.fit(X_train,y_train)
#op_dict['RandomForest'] = clf.score(X_test,y_test)
#average_precision = average_precision_score(y_test, clf.predict(X_test))
#pre_dict["RandomForest_precision"] = average_precision
#
#clf=kn()
#clf.fit(X_train,y_train)
#op_dict['Kneighbours'] = clf.score(X_test,y_test)
#average_precision = average_precision_score(y_test, clf.predict(X_test))
#pre_dict["Kneighbours_precision"] = average_precision
#
#clf=dt()
#clf.fit(X_train,y_train)
#op_dict['DecisionTree'] = clf.score(X_test,y_test)
#average_precision = average_precision_score(y_test, clf.predict(X_test))
#pre_dict["DecisionTree_precision"] = average_precision

print("\nAccuracy Of Models:")
for x,y in op_dict.items():
    print(x,y)
print("\nPrecision Of Models:")    
for x,y in pre_dict.items():    
    print(x,y)
print("\nRecall Of Models:")
for x,y in rec_dict.items():
    print(x,y)
    
