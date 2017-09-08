# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:25:16 2017

@author: 7
"""

import pandas as pd
import numpy as np
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


def log_transform(feature):
    train[feature] = np.log1p(train[feature].values)
    test[feature] = np.log1p(test[feature].values)
    
def text(feature):
    train[feature] = train[feature].apply(lambda x: 1 if x > 0 else 0)
    test[feature] = test[feature].apply(lambda x: 1 if x > 0 else 0)
    

def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))








#train['SaleCondition'] = train['SaleCondition'].apply(lambda x: 1 if x == 'Normal' else 0)
train['SaleCondition'] = train['SaleCondition'].apply(lambda x: 1 if x == 'Normal' else x)
train['SaleCondition'] = train['SaleCondition'].apply(lambda x: 2 if x == 'Abnorml' else 0)
test['SaleCondition'] = test['SaleCondition'].apply(lambda x: 1 if x == 'Normal' else x)
test['SaleCondition'] = test['SaleCondition'].apply(lambda x: 2 if x == 'Abnorml' else 0)

train['SaleType'] = train['SaleType'].apply(lambda x: 1 if x == 'New' else x)
train['SaleType'] = train['SaleType'].apply(lambda x: 2 if x == 'WD' else 0)
test['SaleType'] = test['SaleType'].apply(lambda x: 1 if x == 'New' else x)
test['SaleType'] = test['SaleType'].apply(lambda x: 2 if x == 'WD' else 0)

#train['GarageType'] = train['GarageType'].apply(lambda x: 1 if x == 'BuiltIn' else x)
#train['GarageType'] = train['GarageType'].apply(lambda x: 3 if x == 'Detchd' else x)
#train['GarageType'] = train['GarageType'].apply(lambda x: 2 if x == 'Attchd' else 0)
#test['GarageType'] = test['GarageType'].apply(lambda x: 1 if x == 'BuiltIn' else x)
#test['GarageType'] = test['GarageType'].apply(lambda x: 3 if x == 'Detchd' else x)
#test['GarageType'] = test['GarageType'].apply(lambda x: 2 if x == 'Attchd' else 0)

#train['MiscFeature'] = train['MiscFeature'].apply(lambda x: 1 if x == 'NA' else 0)
#test['MiscFeature'] = test['MiscFeature'].apply(lambda x: 1 if x == 'NA' else 0)

'''
train['HasBasement'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasGarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train['Has2ndFloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasMasVnr'] = train['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
train['HasWoodDeck'] = train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasPorch'] = train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasPool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train['IsNew'] = train['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

test['HasBasement'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasGarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test['Has2ndFloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasMasVnr'] = test['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
test['HasWoodDeck'] = test['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasPorch'] = test['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasPool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test['IsNew'] = test['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

'''
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)
#log_transform('GrLivArea')
#log_transform('1stFlrSF')
#log_transform('2ndFlrSF')
#log_transform('TotalBsmtSF')
#log_transform('LotArea')
#log_transform('LotFrontage')
#log_transform('KitchenAbvGr')
#log_transform('GarageArea')    
#
#text('TotalBsmtSF')
#text('GarageArea')
#text('2ndFlrSF')
#text('MasVnrArea')



#print train['TotalBsmtSF']
#print type(train), type(test)
#print train.shape,test.shape
#print train.columns
#
#fetuercolm=['MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope'\
#,'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle',\
#'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',\
#'BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir'\
#,'Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr'\
#,'KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish'\
#,'GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'\
#,'PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']

fetuercolm=['Id','MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope'\
,'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle',\
'RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',\
'BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir'\
,'Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr'\
,'KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish'\
,'GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'\
,'PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition']


trainfet=train[fetuercolm]
lable=train['SalePrice']

#all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
#
#print "all",type(all_data)


nmiricfeture = train.dtypes[train.dtypes != "object"].index
nmiricfeture=nmiricfeture.drop(['SalePrice','Id','LotFrontage','OverallCond'])

print nmiricfeture
print len(nmiricfeture)
trainfetnum=train[nmiricfeture]
testfetnum=test[nmiricfeture]

#trainfetnum=trainfetnum.drop(['Id'],axis=1)
#testfetnum=testfetnum.drop(['Id'],axis=1)

trainfetnum=trainfetnum.fillna(trainfetnum.mean())
#trainfetnum=trainfetnum.fillna(0)

testfetnum=testfetnum.fillna(testfetnum.mean())
#testfetnum=testfetnum.fillna(0)
#print trainfetnum.head()
trainlist=[trainfetnum]
testlist=[testfetnum]

print "train",type(trainfetnum)



for dataframe in trainlist:
    dataframe['MSSubClass2']=dataframe['MSSubClass']*dataframe['MSSubClass']
    dataframe['LotArea2']=dataframe['LotArea']*dataframe['LotArea']
#    dataframe['LotFrontage2']=dataframe['LotFrontage']*dataframe['LotFrontage']
    dataframe['OverallQual2']=dataframe['OverallQual']*dataframe['OverallQual']
#    dataframe['OverallCond2']=dataframe['OverallCond']*dataframe['OverallCond']
    dataframe['YearBuilt2']=dataframe['YearBuilt']*dataframe['YearBuilt']
    dataframe['YearRemodAdd2']=dataframe['YearRemodAdd']*dataframe['YearRemodAdd']
    dataframe['MasVnrArea2']=dataframe['MasVnrArea']*dataframe['MasVnrArea']    
    
    dataframe['BsmtFinSF12']=dataframe['BsmtFinSF1']*dataframe['BsmtFinSF1']
    dataframe['BsmtFinSF22']=dataframe['BsmtFinSF2']*dataframe['BsmtFinSF2']
    dataframe['BsmtUnfSF2']=dataframe['BsmtUnfSF']*dataframe['BsmtUnfSF']
    dataframe['TotalBsmtSF2']=dataframe['TotalBsmtSF']*dataframe['TotalBsmtSF']
    dataframe['1stFlrSF2']=dataframe['1stFlrSF']*dataframe['1stFlrSF']
    dataframe['2ndFlrSF2']=dataframe['2ndFlrSF']*dataframe['2ndFlrSF']
    dataframe['LowQualFinSF2']=dataframe['LowQualFinSF']*dataframe['LowQualFinSF']
    dataframe['GrLivArea2']=dataframe['GrLivArea']*dataframe['GrLivArea'] 
    
    dataframe['BsmtFullBath2']=dataframe['BsmtFullBath']*dataframe['BsmtFullBath'] 
    dataframe['BsmtHalfBath2']=dataframe['BsmtHalfBath']*dataframe['BsmtHalfBath'] 
    dataframe['FullBath2']=dataframe['FullBath']*dataframe['FullBath'] 
    dataframe['HalfBath2']=dataframe['HalfBath']*dataframe['HalfBath'] 
    dataframe['BedroomAbvGr2']=dataframe['BedroomAbvGr']*dataframe['BedroomAbvGr'] 
    dataframe['KitchenAbvGr2']=dataframe['KitchenAbvGr']*dataframe['KitchenAbvGr'] 
    dataframe['TotRmsAbvGrd2']=dataframe['TotRmsAbvGrd']*dataframe['TotRmsAbvGrd'] 
    dataframe['Fireplaces2']=dataframe['Fireplaces']*dataframe['Fireplaces'] 
    dataframe['GarageYrBlt2']=dataframe['GarageYrBlt']*dataframe['GarageYrBlt'] 
    
    dataframe['GarageCars2']=dataframe['GarageCars']*dataframe['GarageCars'] 
    dataframe['GarageArea2']=dataframe['GarageArea']*dataframe['GarageArea'] 
    dataframe['WoodDeckSF2']=dataframe['WoodDeckSF']*dataframe['WoodDeckSF'] 
    dataframe['OpenPorchSF2']=dataframe['OpenPorchSF']*dataframe['OpenPorchSF'] 
    dataframe['EnclosedPorch2']=dataframe['EnclosedPorch']*dataframe['EnclosedPorch'] 
    dataframe['3SsnPorch2']=dataframe['3SsnPorch']*dataframe['3SsnPorch'] 
    dataframe['ScreenPorch2']=dataframe['ScreenPorch']*dataframe['ScreenPorch'] 
    dataframe['PoolArea2']=dataframe['PoolArea']*dataframe['PoolArea'] 
    dataframe['MiscVal2']=dataframe['MiscVal']*dataframe['MiscVal'] 
    dataframe['MoSold2']=dataframe['MoSold']*dataframe['MoSold'] 
    dataframe['YrSold2']=dataframe['YrSold']*dataframe['YrSold'] 
    

    
for dataframe in testlist:
    dataframe['MSSubClass2']=dataframe['MSSubClass']*dataframe['MSSubClass']
    dataframe['LotArea2']=dataframe['LotArea']*dataframe['LotArea']
#    dataframe['LotFrontage2']=dataframe['LotFrontage']*dataframe['LotFrontage']
    dataframe['OverallQual2']=dataframe['OverallQual']*dataframe['OverallQual']
#    dataframe['OverallCond2']=dataframe['OverallCond']*dataframe['OverallCond']
    dataframe['YearBuilt2']=dataframe['YearBuilt']*dataframe['YearBuilt']
    dataframe['YearRemodAdd2']=dataframe['YearRemodAdd']*dataframe['YearRemodAdd']
    dataframe['MasVnrArea2']=dataframe['MasVnrArea']*dataframe['MasVnrArea']
    
    dataframe['BsmtFinSF12']=dataframe['BsmtFinSF1']*dataframe['BsmtFinSF1']
    dataframe['BsmtFinSF22']=dataframe['BsmtFinSF2']*dataframe['BsmtFinSF2']
    dataframe['BsmtUnfSF2']=dataframe['BsmtUnfSF']*dataframe['BsmtUnfSF']
    dataframe['TotalBsmtSF2']=dataframe['TotalBsmtSF']*dataframe['TotalBsmtSF']
    dataframe['1stFlrSF2']=dataframe['1stFlrSF']*dataframe['1stFlrSF']
    dataframe['2ndFlrSF2']=dataframe['2ndFlrSF']*dataframe['2ndFlrSF']
    dataframe['LowQualFinSF2']=dataframe['LowQualFinSF']*dataframe['LowQualFinSF']
    dataframe['GrLivArea2']=dataframe['GrLivArea']*dataframe['GrLivArea'] 
    
    dataframe['BsmtFullBath2']=dataframe['BsmtFullBath']*dataframe['BsmtFullBath'] 
    dataframe['BsmtHalfBath2']=dataframe['BsmtHalfBath']*dataframe['BsmtHalfBath'] 
    dataframe['FullBath2']=dataframe['FullBath']*dataframe['FullBath'] 
    dataframe['HalfBath2']=dataframe['HalfBath']*dataframe['HalfBath'] 
    dataframe['BedroomAbvGr2']=dataframe['BedroomAbvGr']*dataframe['BedroomAbvGr'] 
    dataframe['KitchenAbvGr2']=dataframe['KitchenAbvGr']*dataframe['KitchenAbvGr'] 
    dataframe['TotRmsAbvGrd2']=dataframe['TotRmsAbvGrd']*dataframe['TotRmsAbvGrd'] 
    dataframe['Fireplaces2']=dataframe['Fireplaces']*dataframe['Fireplaces'] 
    dataframe['GarageYrBlt2']=dataframe['GarageYrBlt']*dataframe['GarageYrBlt'] 
    
    dataframe['GarageCars2']=dataframe['GarageCars']*dataframe['GarageCars'] 
    dataframe['GarageArea2']=dataframe['GarageArea']*dataframe['GarageArea'] 
    dataframe['WoodDeckSF2']=dataframe['WoodDeckSF']*dataframe['WoodDeckSF'] 
    dataframe['OpenPorchSF2']=dataframe['OpenPorchSF']*dataframe['OpenPorchSF'] 
    dataframe['EnclosedPorch2']=dataframe['EnclosedPorch']*dataframe['EnclosedPorch'] 
    dataframe['3SsnPorch2']=dataframe['3SsnPorch']*dataframe['3SsnPorch'] 
    dataframe['ScreenPorch2']=dataframe['ScreenPorch']*dataframe['ScreenPorch'] 
    dataframe['PoolArea2']=dataframe['PoolArea']*dataframe['PoolArea'] 
    dataframe['MiscVal2']=dataframe['MiscVal']*dataframe['MiscVal'] 
    dataframe['MoSold2']=dataframe['MoSold']*dataframe['MoSold'] 
    dataframe['YrSold2']=dataframe['YrSold']*dataframe['YrSold'] 
    


print trainfetnum.shape
print testfetnum.shape


from sklearn.cross_validation import train_test_split,cross_val_score



features_train, features_test, labels_train, labels_test = \
    train_test_split(trainfetnum, lable, test_size=0.3, random_state=42)




from sklearn.linear_model import LinearRegression,LassoLarsCV
#reg = LinearRegression()
#reg.fit(features_train,labels_train)

reg=LassoLarsCV(max_iter=10000)
reg.fit(features_train,labels_train)
#score=cross_val_score(reg,features_train,labels_train,cv=10,scoring='accuracy')
#score.fit(features_train,labels_train)
#print score


#print reg.score(features_train, labels_train)
#print reg.score(features_test, labels_test)
#print reg.coef_
#print reg.intercept_

pre = reg.predict(features_test)
print labels_test.shape
print features_test.shape


#from sklearn.metrics import mean_squared_error

print "errror2",error(labels_test,pre)
pre1=pre+3000
print "errror2",error(labels_test,pre1)

#print type(trainfet) , trainfet.shape , train.shape
#print type(lable) , lable.shape 
#          
#trainlist=[trainfet]
#print type(trainlist)
#///////////////////////////////////////////////////////

import matplotlib.pyplot as plt
x=features_train['OverallQual']
plt.scatter(x, labels_train,  color='r')
plt.plot(x, reg.predict(features_train), color='blue',linewidth=3)
plt.show()




pre = reg.predict(testfetnum)
pre=pre+3000

print pre

df = pd.DataFrame({'SalePrice' : pre,'Id' : test['Id']})
df.to_csv('results.csv', header=True ,index=False)

