# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 06:58:27 2017

@author: 7
"""


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt, matplotlib.image as mpimg

#def standardize(x): 
#    return (x-mean_px)/std_px


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

feture = train.iloc[0:,1:]
lable = train.iloc[0:,:1]

feturelist=[feture]
lablelist=[lable]

testl=[test]
print feture.shape
for f in feturelist:
    f['mean']=f.mean(axis=1)
    
for t in testl:
    t['mean']=t.mean(axis=1)
#    f['std']=f.std(axis=1)
print feture.shape
#ftrain=[feture]

#print type(ftrain)

#newtrain=feture.mean(axis=1)
#print type(newtrain)
#print newtrain.shape
#newtrain = newtrain[:, np.newaxis]
#print newtrain.shape
#print newtrain

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(feture, lable, test_size=0, random_state=42)
    
#print type(labels_train)

#print len(labels_train)
#print features_train

features_train[features_train>0]=1
#features_test[features_test>0]=1
    
    
#from keras.models import  Sequential
#from keras.layers.core import  Lambda , Dense, Flatten, Dropout
#from keras.callbacks import EarlyStopping
#from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D
#
#model= Sequential()
#model.add(Lambda(standardize,input_shape=(28,28,1)))
#model.add(Flatten())
#model.add(Dense(10, activation='softmax'))
#print("input shape ",model.input_shape)
#print("output shape ",model.output_shape)



#from sklearn import svm
#clf = svm.SVC()

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=10)
#
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=170)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

#clf = clf.fit(features_train, labels_train)
clf = clf.fit(features_train, labels_train.values.ravel())
#

print clf.score(features_train, labels_train)
#print clf.score(features_test, labels_test)

pre = clf.predict(test)

df = pd.DataFrame(pre)
df.index.name='ImageId'
df.index+=1
df.columns=['ImageId']

df.to_csv('results.csv', header=True)

#ImageId,Label
