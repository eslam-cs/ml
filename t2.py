# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:02:41 2017

@author: 7
"""

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt




def age(i):
    if i <= 16 or i=='':
        return 0
    elif i<=32:
        return 1
    elif i<=48:
        return 2
    elif i<=64:
        return 3
    else:
        return 4
def alone(i):
    if i > 0:
        return 1
    else:
        return 0
#def cabin(i):
#    if i.item():
#        print type(i)
#        return 0
#    else:
#        return 1
    
    
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


print train.columns.values

#print train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False)
#print train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().sort_values(by='Survived', ascending=False)
#print train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False)
#print train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=True).mean().sort_values(by='Survived', ascending=False)
#print train[['Parch', 'Survived']].groupby(['Parch'], as_index=True).mean().sort_values(by='Survived', ascending=False)


#g = sns.FacetGrid(train, col='Survived')
#g.map(plt.hist, 'Age', bins=20)

print "shape" , train.shape , test.shape

train=train.drop(['Ticket','Cabin'],axis=1)
test=test.drop(['Ticket','Cabin'],axis=1)
combin=[train,test]
trainlist=[train]
testlist=[test]


print "djhdjshjdhsjd",type(trainlist)
print "shape" , train.shape , test.shape 

for dataset in trainlist:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#    dataset.loc[dataset['SibSp']+dataset['Parch'] > 0, 'IsAlone'] = 0
#    dataset.loc[dataset['SibSp']+dataset['Parch'] == 0, 'IsAlone'] = 1
    dataset['IsAlone']=dataset['SibSp']+dataset['Parch']
    dataset.loc[ dataset['Age'] <= 16 , 'Age'] = 0
    dataset['Age']=dataset['Age'].fillna(0)
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']    
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, np.NaN : 3} ).astype(int)   
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['AgeClass']=dataset['Age']*dataset['Pclass'].astype(int)
    
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)+1
           

#    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: 1 if not pd.isnull(x) else 0)
#    dataset['Cabin'] = dataset['Cabin'].fillna(0)

    #myyyyyyyyyy.
    dataset['titlesex']=dataset['Title']*dataset['Sex']
#    dataset['pclasssex']=dataset['Pclass']*dataset['titlesex']
#    dataset['all']=dataset['Title']*dataset['Sex']*dataset['IsAlone']*dataset['AgeClass']*dataset['Age']
    
    
    
for dataset in testlist:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#    dataset.loc[dataset['SibSp']+dataset['Parch'] > 0, 'IsAlone'] = 0
#    dataset.loc[dataset['SibSp']+dataset['Parch'] == 0, 'IsAlone'] = 1
    dataset['IsAlone']=dataset['SibSp']+dataset['Parch']+1
    dataset.loc[ dataset['Age'] <= 16 , 'Age'] = 0
    dataset['Age']=dataset['Age'].fillna(0)
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']    
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, np.NaN : 3} ).astype(int)  
    dataset['Fare']=dataset['Fare'].fillna(0)
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['AgeClass']=dataset['Age']*dataset['Pclass'].astype(int)
    
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)+1

#    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: 1 if not pd.isnull(x) else 0)
#    dataset['Cabin'] = dataset['Cabin'].fillna(0)
    #myyyyyyyy
    dataset['titlesex']=dataset['Title']*dataset['Sex']
#    dataset['pclasssex']=dataset['Pclass']*dataset['titlesex']
#    dataset['all']=dataset['Title']*dataset['Sex']*dataset['IsAlone']*dataset['AgeClass']*dataset['Age']
    
#    dataset['Age'] = dataset['Age'].map({np.NaN : 0})
#    dataset['Embarked'] = dataset['Embarked'].map( {'S': 1, 'Q': 2, 'C': 0} ).astype(int)
print type(trainlist)
#for dataset in trainlist:
#    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#pd.crosstab(train['Title'], train['Sex'])

lable=train['Survived']
passinger=test['PassengerId']

train=train.drop(['Parch', 'SibSp','PassengerId','Survived','Name','Sex','Pclass'],axis=1)
test=test.drop(['Parch', 'SibSp','PassengerId','Name','Sex','Pclass'],axis=1)
print train.head(15)




from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(train, lable, test_size=0, random_state=42)


from sklearn import svm
clf = svm.SVC()

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=10)
#
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier()

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#print len(features_train['titlesex']) ,len(labels_train) ,len(features_train)
clf = clf.fit(features_train, labels_train)

print clf.score(features_train, labels_train)
#print clf.score(features_test, labels_test)

pre = clf.predict(test)



goal=[]
for t,p in zip(passinger,pre):
    g=[]
    g.append(t)  
    g.append(p)
    goal.append(g)

#print type(testlist)

head="PassengerId,Survived"
import csv
with open('goal.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow([head])
    for g in goal:
        wr.writerow(g)




