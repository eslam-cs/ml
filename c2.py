# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:25:16 2017

@author: 7
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
#train=pd.read_csv('train.csv', header=None)
print train.shape,test.shape
print test.columns
#from math import radians, cos, sin, asin, sqrt


def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine(lat1, lng1, lat1, lng2)
    b = haversine(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
#    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))



print train.columns
#print train.head()


train.loc[:, 'distance_haversine'] = haversine(train['pickup_latitude'].values,train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine(test['pickup_latitude'].values,test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train.loc[:, 'distance_haversine2']=train['distance_haversine']*train['distance_haversine']
test.loc[:, 'distance_haversine2']=test['distance_haversine']*test['distance_haversine']


train.loc[:, 'dummy_manhattan_distance'] = dummy_manhattan_distance(train['pickup_latitude'].values,train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'dummy_manhattan_distance'] = dummy_manhattan_distance(test['pickup_latitude'].values,test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
#
train.loc[:, 'dummy_manhattan_distance2'] = train['dummy_manhattan_distance']*train['dummy_manhattan_distance']
test.loc[:, 'dummy_manhattan_distance2'] = test['dummy_manhattan_distance']*test['dummy_manhattan_distance']
#
train.loc[:, 'bearing_array'] = bearing_array(train['pickup_latitude'].values,train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'bearing_array'] = bearing_array(test['pickup_latitude'].values,test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
#
train.loc[:, 'bearing_array2']=train['bearing_array']*train['bearing_array']
test.loc[:, 'bearing_array2']=test['bearing_array']*test['bearing_array']

train.loc[:, 'location']=train['pickup_latitude']+train['pickup_longitude']
test.loc[:, 'location']=test['pickup_latitude']+test['pickup_longitude']
#
train.loc[:, 'location2']=train['location']*train['location']
test.loc[:, 'location2']=test['location']*test['location']

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
#
train.loc[:, 'month']=train['pickup_datetime'].dt.month
test.loc[:, 'month']=test['pickup_datetime'].dt.month
        

        
train.loc[:, 'hour']=train['pickup_datetime'].dt.hour
test.loc[:, 'hour']=test['pickup_datetime'].dt.hour
        
train.loc[:, 'hour']=train['pickup_datetime'].dt.weekday
test.loc[:, 'hour']=test['pickup_datetime'].dt.weekday

#dataframe['mintdrob']=dataframe['dropoff_datetime'].dt.minute
#    dataframe['hourdrob']=dataframe['dropoff_datetime'].dt.hour
#    dataframe['daydrob']=dataframe['dropoff_datetime'].dt.weekday
#    dataframe['yeardrob']=dataframe['dropoff_datetime'].dt.weekofyear


nmiricfeture = train.dtypes[train.dtypes != "object"].index
print nmiricfeture                 


plt.figure()
plt.scatter(train['pickup_longitude'].values[:10000],train['pickup_latitude'].values[:10000],c='b',s=1)
plt.show()
plt.figure()
plt.scatter(train['dropoff_longitude'].values[:10000],train['dropoff_latitude'].values[:10000],c='g',s=1)
plt.show()

corr = train.corr()
corr.sort_values(["trip_duration"], ascending = False, inplace = True)
print(corr.trip_duration)

lable=train['trip_duration']

nmiricfeture=nmiricfeture.drop(['vendor_id','passenger_count','pickup_longitude','pickup_latitude'\
                                ,'dropoff_longitude','dropoff_latitude','pickup_datetime'])
    
print nmiricfeture  

#from pandas.tools.plotting import parallel_coordinates
#plt.figure()
#parallel_coordinates(train[nmiricfeture], 'trip_duration')
#plt.show()

nmiricfeture=nmiricfeture.drop(['trip_duration'])
traindone=train[nmiricfeture]
testdone=test[nmiricfeture]

print traindone.shape

from sklearn.cluster import KMeans
model = KMeans(7)
model.fit(traindone)
knntrain = model.predict(traindone)
knntest = model.predict(testdone)

print len(knntrain)
traindone.loc[:, 'knn']=knntrain
testdone.loc[:, 'knn']=knntest
         
#train['knn']=knntrain
#test['knn']=knntrain

from sklearn import preprocessing
traindone= preprocessing.normalize(traindone, axis=0)
testdone= preprocessing.normalize(testdone, axis=0)
          
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(traindone)
traindone = pca.transform(traindone)

pca.fit(testdone)
testdone = pca.transform(testdone)

#print traindone.shape
#print traindone.head()



from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(traindone, lable, test_size=0.3, random_state=42)
    
    
from sklearn.linear_model import LinearRegression,LassoLarsCV, Lasso
reg=Lasso(alpha = 0.01)
reg.fit(features_train,labels_train)

#reg = LinearRegression()
#reg.fit(features_train,labels_train)
#
#reg=LassoLarsCV(max_iter=100000)
#reg.fit(features_train,labels_train)


print reg.score(features_train, labels_train)
print reg.score(features_test, labels_test)
print reg.coef_
print reg.intercept_

pre = reg.predict(features_test)
print labels_test.shape
print features_test.shape
pre=abs(pre)
print "errror",error(labels_test,pre)



#for i in nmiricfeture: 
#tit= 'distance_haversine2'
#x=features_train[tit]
#plt.scatter(x, labels_train,  color='r')
#plt.plot(x, reg.predict(features_train), color='blue',linewidth=3)
#plt.title(tit)
#plt.show()
#
#x=features_test[tit]
#plt.scatter(x, labels_test,  color='r')
#plt.plot(x, pre, color='blue',linewidth=3)
#plt.title(tit)
#plt.show()

pre1 =pre/1.36
pre1=abs(pre1)
print "errror1",error(labels_test,pre1)


#plt.scatter(x, labels_test,  color='r')
#plt.plot(x, pre1, color='blue',linewidth=3)
#plt.title(tit)
#plt.show()

#print test.columns
#
#pre2 =pre/1.34
#print "errror2",error(labels_test,pre2)
#pre3 =pre/1.35
#print "errror3",error(labels_test,pre3)
#pre4 =pre/1.37
#print "errror4",error(labels_test,pre4)

pre = reg.predict(testdone)
pre=abs(pre)
pre =pre/1.36
#
#
df = pd.DataFrame({'trip_duration' : pre,'Id' : test['id']})
df.to_csv('results.csv', header=True ,index=False)
