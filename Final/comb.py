import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib as mp
from scipy.stats import skew
import csv
import math
from sklearn import linear_model
import xgboost as xgb

from sklearn.linear_model import Ridge,RidgeCV,ElasticNet,LassoCV,LassoLarsCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor as rfr

data_train=[]
data_test=[]
data_final=[]
price=[]


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, data_train, price, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

test=pd.read_csv('../input/test.csv')	
train=pd.read_csv('../input/train.csv')


data_final = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

train["SalePrice"] = np.log1p(train["SalePrice"])

numeric_feats = data_final.dtypes[data_final.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
data_final[skewed_feats] = np.log1p(data_final[skewed_feats])


data_final = pd.get_dummies(data_final)

data_final = data_final.fillna(data_final[:train.shape[0]].mean())

data_train = data_final[:train.shape[0]]
data_test = data_final[train.shape[0]:]

price = train.SalePrice


rf_model=rfr(n_estimators=100)
rf_model.fit(data_train,price)
rmse1=np.sqrt(-cross_val_score(rf_model,data_train,price,scoring="neg_mean_squared_error",cv=5))


lasso_model = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 5e-4]).fit(data_train, price)
rmse2=rmse_cv(lasso_model)


dtrain = xgb.DMatrix(data_train, label = price)
dtest = xgb.DMatrix(data_test)
params = {"max_depth":2, "eta":0.09}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

xgb_model=xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0.045,                 
                 learning_rate=0.01,
                 max_depth=20,
                 min_child_weight=1.5,
                 n_estimators=6000,                                                                    
                 reg_alpha=0.65,
                 reg_lambda=0.45,
                 subsample=0.75)


#xgb_model = xgb.XGBRegressor(n_estimators = 6000, seed = 0, learning_rate = 0.01, max_depth = 3, subsample = 0.8, 
 #                            colsample_bytree = 0.8, colsample_bylevel = 0.8 )
xgb_model.fit(data_train, price)


rf_preds = np.expm1(rf_model.predict(data_test))
lasso_preds = np.expm1(lasso_model.predict(data_test))
xgb_preds=np.expm1(xgb_model.predict(data_test))

final_result=0.8*lasso_preds+0.1*xgb_preds+0.1*rf_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":final_result}, columns=['id', 'SalePrice'])
solution.to_csv("comb2.csv", index = False)