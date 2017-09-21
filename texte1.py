# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 00:20:57 2017

@author: 7
"""
import re
import pandas as pd
from num2words import num2words

SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
OTH = str.maketrans("፬", "4")
clen=str.maketrans(","," ")
clen2=str.maketrans("-"," ")


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def mapping(x):
    
    if x in d.keys():
        return d[x]
    elif str(x).isdigit():
        x = x.translate(SUB)
        x = x.translate(SUP)
        x = x.translate(OTH)
        x = x.translate(clen)
        x = x.translate(clen2)
        x=num2words(float(x)) 
        return x

    elif hasNumbers(str(x)):
        print(x)
        x = x.translate(SUB)
        x = x.translate(SUP)
        x = x.translate(OTH)
        x = x.translate(clen2)
        x = re.sub(",", "", x)
        lx=x.split()  
        dx=""
        for i in lx:
            if i in sdict.keys():
                dx += sdict[i] 
                dx += " "
            elif str(i).isdigit():
                i = num2words(float(i))
                dx += i
                dx += " "
            elif re.match(".*\d+\.\d+.*", i):
#                i = re.sub(",", "", i)
                flag=re.findall("\d+\.\d+", i)
                for m in flag:
                    rep=num2words(float(m))
#                    out=re.sub(m,rep, i)
                    dx += rep
                    dx += " "
            elif re.match(".*\d+:\d+.*", i):
                i = re.sub(":", " ", i)
                flag=re.findall("\d+", i)
                for m in flag:
                    rep=num2words(float(m))
                    dx += rep
                    dx += " "
            elif re.match(".*\d+.*", i):
#                
#                i = re.sub(",", "", i)
                flag=re.findall("\d+", i)
                for m in flag:
                    rep=num2words(float(m))
                    dx += rep
                    dx += " "
            else:
                dx += i
                dx += " "
        print(dx)
        return dx
#    elif hasNumbers(str(x)):
#        x = x.translate(SUB)
#        x = x.translate(SUP)
#        x = x.translate(OTH)
#        if re.match(".*\d+\.\d+.*", x):
#            print(x)
#            x = re.sub(",", "", x)
#            flag=re.findall("\d+\.\d+", x)
#
#            for i in flag:
#                rep=num2words(float(i))
#                out=re.sub(i,rep, x)
#            print(out)
#            return out
#        elif re.match(".*\D*.*", x):
#            print(x)
#            x = re.sub(",", "", x)
#            flag=re.sub("\D*", "", x)
#            rep=num2words(flag)
#            out=re.sub(flag,rep, x)
#            print(out)
#            return out
#        else:
#            return x
    else:
        return x
    

    
sdict = {}
sdict['km2'] = 'square kilometers'
sdict['km'] = 'kilometers'
sdict['kg'] = 'kilograms'
sdict['lb'] = 'pounds'
sdict['dr'] = 'doctor'
sdict['m²'] = 'square meters'
sdict['pm'] = 'p m' 




train = pd.read_csv('en_train.csv')
test = pd.read_csv('en_test.csv')


print(train.describe())
print(train.dtypes)
print(train.head())
print(train['class'].unique())
print(train['class'].value_counts())

t1=train.loc[train['class']=='DATE']
print(t1[['before','after']].head(20))

d = train.groupby(['before', 'after']).size()
d = d.reset_index().sort_values(0, ascending=False)
d = d.loc[d['before'].drop_duplicates(keep='first').index]
d = d.loc[d['before'] != d['after']]
d = d.set_index('before')['after'].to_dict()

#print(d)

    
test['after'] =  test.before.apply(mapping) 


test['id'] = test.sentence_id.astype(str) + '_' + test.token_id.astype(str)
test[['id', 'after']].to_csv('res.csv', index=False)

