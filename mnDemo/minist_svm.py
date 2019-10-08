# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:10:01 2018

@author: jhdn
"""
from sklearn.metrics import accuracy_score
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from sklearn.svm import LinearSVC 
from sklearn.cross_validation  import train_test_split
#from sklearn.grid_search import GridSearchCV

#读数据
train_data=pd.read_csv('./train.csv')
test_data=pd.read_csv('./test.csv')

#将标签分离
label=train_data['label']
train_data.drop('label',axis=1,inplace=True)



#交叉验证
x_train,x_test,y_train,y_test=train_test_split(train_data,label,test_size=0.3)
clf=LinearSVC(verbose=2)
clf.fit(x_train,y_train)
pre=clf.predict(x_test)
print(accuracy_score(y_test,pre))

#网格调参
#param_grid={'C':[1,0.5,0.3,0.01]}
#grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
#grid.fit(train_data,label)

#预测
clf.fit(train_data,label)
result=df(clf.predict(test_data),index=range(1,28001))

result.to_csv("result.csv",header=['Label'],index_label='ImageId')


