import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
from keras.models import Sequential
from keras.utils import np_utils  

sns.set()
weight_name='demo_weights.h5'
exist_weight=os.path.exists(weight_name)
'''
数据导入

'''
#train_data=pd.read_csv(r'train.csv',nrows=1000)
#test_data=pd.read_csv(r'test.csv',nrows=1000)
train_data=pd.read_csv(r'train.csv')
test_data=pd.read_csv(r'test.csv')

'''
查看数据分布
一张,讲解Python中的形状(n,)与(n,m)的坑与区别
'''
def visaulize(data):
    image=data[:784].values.reshape(1,-1).reshape(28,28)
    plt.imshow(image, cmap='gray')
    plt.show()
#for i in range(10):
#    visaulize(train_data.iloc[i])

'''
数据预处理（异常值,归一化,值填充）
只有一个转换形状
'''

def preprocessing(data,names):
    return data[names].copy()

def OneHot(data):
    data=np_utils.to_categorical(data,num_classes=10)
    return data

#取出标签
labels=train_data['label']
train_data.drop(['label'],axis=1,inplace=True)

labels=OneHot(labels)
#归一化非常重要
train_data=train_data/255
test_data=train_data/255
'''
特征工程（造特征，筛选特征）
'''
train_data=train_data.values.reshape(-1,1,28,28)
test_data=test_data.values.reshape(-1,1,28,28)

'''
切分数据+训练+线下验证分数（训练集）
'''
train_X,test_X,trainY,test_Y=train_test_split(train_data,labels,test_size=0.3)
def CNN_model():
    model = Sequential()    
    
    model.add(Conv2D(activation="selu", input_shape=(1, 28, 28), padding="same", kernel_size=(5, 5), data_format="channels_first", filters=32))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64,(5,5),activation='selu'))    
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))    
    model.add(Dropout(0.3))    
    
    model.add(Flatten()) 
    model.add(Dense(1024,activation='selu'))    
   
    model.add(Dense(128,activation='selu'))      
    model.add(Dense(10,activation='softmax'))    
    #Compile model    
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])    
    return model   


model=CNN_model()
if not exist_weight: 
    model.fit(train_X,trainY,epochs=5,batch_size=64) 
else:
    model.load_weights(weight_name)

score=model.evaluate(test_X,test_Y)
print(score)
model.save_weights(weight_name)
'''
重新训练,保存+提交结果（测试集）
利用test_data_01.info()发现Fare属性有空值
因为时间原因就不训练了

model2=CNN_model()
model2.fit(train_data,labels)
predict=model2.predict(test_data)

'''
#注意这里是独热码，要转回数字结果
predict=model.predict(test_data).argmax(1)  


result=pd.DataFrame()
result['ImageId']=range(len(predict))
result['ImageId']=result['ImageId']+1
result['Label']=predict
result.to_csv(r'result.csv')