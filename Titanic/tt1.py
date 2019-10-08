
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
'''
数据导入
'''
train_data=pd.read_csv(r'train.csv')
test_data=pd.read_csv(r'test.csv')

'''
数据预处理（异常值,归一化,值填充）
'''
def preprocessing(data,names):
    return data[names].copy()
#取出标签
labels=train_data['Survived']

train_data_01=preprocessing(train_data,['Sex','SibSp','Parch','Fare','Survived'])
test_data_01=preprocessing(test_data,['Sex','SibSp','Parch','Fare'])
#对Nan值的填充
train_data_01['Age']=train_data['Age'].copy().fillna(train_data['Age'].mean())
test_data_01['Age']=test_data['Age'].copy().fillna(test_data['Age'].mean())

#对Sex字符串的处理
train_data_01['Sex']=train_data_01['Sex'].copy().apply(lambda x:0 if x=='male' else 1)
test_data_01['Sex']=test_data_01['Sex'].copy().apply(lambda x:0 if x=='male' else 1)


'''
查看数据分布
'''
print(train_data_01.columns.values)
#全部图
sns.pairplot(train_data_01[['Sex','Survived']],hue='Survived',size=2,vars=['Sex'])
#柱状图
sns.countplot(x = "SibSp", hue = "Survived", data = train_data_01)
#散点图
sns.jointplot(x="Age", y="Parch", data=train_data_01, size=4)
#训练集丢掉标签
train_data_01=train_data_01.copy().drop(['Survived'],axis=1)

'''
特征工程（造特征，筛选特征）
'''
train_data_01['new_feature1']=train_data_01['Fare'].copy()*train_data_01['Age'].copy()
train_data_01['new_feature2']=train_data_01['Fare'].copy()**2
train_data_01['new_feature3']=np.log(train_data_01['Fare'].copy().values+1)

#发现test的Fare有nan值，所以填充平均值
test_data_01['Fare']=test_data_01['Fare'].copy().fillna(test_data_01['Fare'].mean())


test_data_01['new_feature1']=test_data_01['Fare'].copy()*test_data_01['Age'].copy()
test_data_01['new_feature2']=test_data_01['Fare'].copy()**2
test_data_01['new_feature3']=np.log(test_data_01['Fare'].copy().values+1)
'''
切分数据+训练+线下验证分数（训练集）
'''
train_X,test_X,trainY,test_Y=train_test_split(train_data_01,labels,test_size=0.3)
LR=LogisticRegression()
LR.fit(train_X,trainY)

ypre=LR.predict(test_X)
print(accuracy_score(test_Y,ypre))
'''
重新训练,保存+提交结果（测试集）
利用test_data_01.info()发现Fare属性有空值
'''
LR.fit(train_data_01,labels)
test_result=LR.predict(test_data_01)
df=pd.DataFrame()
df['PassengerId']=test_data['PassengerId']
df['Survived']=test_result

df.to_csv(r'result.csv',index=None)