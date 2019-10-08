# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:01:51 2018

@author: Hhuafei
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

train=pd.read_csv(r'./train.csv')
test=pd.read_csv(r'./test.csv')
'''
object类型
数字类型
['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
       'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
       'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
       'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
       'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
       'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold',
       'YrSold', 'SaleType', 'SaleCondition', 'SalePrice'(标签)]
'''
print(train.info())
print(train.head())
#打印缺失值的个数
'''
一起处理
不推荐，不符合实际,测试集不应该放在这上面（不可见分布）
首先，我将通过记录log(feature + 1) 来改变倾斜的数字特征 - 这将使特征更加正常
为分类特征创建虚拟变量
将数字缺失值（NaN's）替换为各自列的平均值
'''
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
'''
y=log1p(x)
x=np.exp(y)-1
转换，使大致符合正态分布
'''
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
#转换
train["SalePrice"] = np.log1p(train["SalePrice"])
#非字符串类型
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#偏度
#准备将正偏的数据转变一下
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
'''
粗暴，。。自动对非数字型数据one_hot编码，数值型数据保留，one-hot编码适合类别型而不是大小型数据
all_data.mean()自动对每一列的数据求平均值
'''
all_data=pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
'''
切分回来
'''
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice
'''
模型选择，训练
'''
from sklearn.linear_model import Ridge,LassoCV
from sklearn.model_selection import cross_val_score
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_absolute_error", cv = 5))
    return(rmse)
    
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "cost mse")
plt.xlabel("alpha")
plt.ylabel("rmse")
'''
LassoCV，传入一堆参数，自动选择α
'''
model_lasso=LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(X_train, y)

coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#前十个和后十个特征的强弱,288个特征太多了
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
'''
最重要的积极特征是GrLivArea - 以平方英尺为单位的上述地面面积。
 这绝对有意义。 然后，其他一些位置和质量特征也有积极作用。 
 一些负面特征的意义不大，并且值得更多关注 - 看起来他们可能来自不平衡的分类变量。
最后再检测结果和预测值的差别
 '''
preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")

solution = pd.DataFrame()
solution["Id"]=test.Id
solution['SalePrice']=np.exp(preds.preds)-1

solution.to_csv("result.csv", index = False)
