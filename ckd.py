#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:05:01 2019

@author: geoffrey.kip
"""
import pandas as pd
import numpy as np
from os import chdir
from ludwig import LudwigModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

wd = "/Users/geoffrey.kip/Projects/ludwig/data/"
chdir(wd)

df = pd.read_csv("kidney_disease.csv")
df = shuffle(df)
df.head()

#Exploratory analysis
print(df.columns)
print(df.dtypes)
print(df.describe)
print(df.isnull().sum())

#reformat outcome
df["classification"] = np.where(df["classification"] == "ckd" , 1, 0)
df.drop(['id'],axis=1 ,inplace=True)

#Split data into training and test sets
train_df = df.iloc[:300,:]
test_df = df[~df.index.isin(train_df.index)]

# Split labels and features for test set
Y_test = test_df.iloc[:,-1]
X_test = test_df.loc[:, train_df.columns != 'classification']


# Run ludwig
model_definition = {
    'input_features':[
        {'name':'age', 'type':'numerical'},
        {'name':'bp', 'type':'numerical'},
        {'name':'sg', 'type':'numerical'},
        {'name':'al', 'type':'numerical'},
        {'name':'su', 'type':'numerical'},
        {'name':'rbc', 'type':'category'},
        {'name':'pc', 'type':'category'},
        {'name':'pcc', 'type':'category'},
        {'name':'ba', 'type':'category'},
        {'name':'bgr', 'type':'numerical'},
        {'name':'bu', 'type':'numerical'},
        {'name':'sc', 'type':'numerical'},
        {'name':'sod', 'type':'numerical'},
        {'name':'pot', 'type':'numerical'},
        {'name':'hemo', 'type':'numerical'},
        {'name':'pcv', 'type':'category'},
        {'name':'wc', 'type':'category'},
        {'name':'rc', 'type':'category'},
        {'name':'htn', 'type':'category'},
        {'name':'dm', 'type':'category'},
        {'name':'cad', 'type':'category'},
        {'name':'appet', 'type':'category'},
        {'name':'pe', 'type':'category'},
        {'name':'ane', 'type':'category'},
     ],
    'output_features': [
        {'name': 'classification', 'type': 'binary'}
    ]
}

print('creating model')
model = LudwigModel(model_definition)
print('training model')
train_stats = model.train(data_df=train_df)

#Run predictions
predictions = model.predict(data_df=X_test)
predictions["classification_predictions"] = np.where(predictions.iloc[:,0] == True,1,0)

print(accuracy_score(Y_test, predictions.iloc[:,0]))
print(confusion_matrix(Y_test, predictions.iloc[:,0]))
print(classification_report(Y_test, predictions.iloc[:,0]))

model.close()