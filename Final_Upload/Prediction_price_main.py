#All the models are tarined in this file using XGboost,
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from operator import itemgetter
import itertools
import xgboost as xgb


def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
lasso_model = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 5e-4]).fit(X_train, y)
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)  
subsample,colsample_bytree,eta,max_depth,num_boost_round,early_stopping_rounds,test_size = 0.8,0.8,0.2,8,400,10,0.2 

print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
params = {
    "objective": "reg:linear",
    "booster" : "gbtree", 
    "eval_metric": "rmse", # this is the metric for the leardboard
    "eta": eta, # shrinking parameters to prevent overfitting
    "tree_method": 'exact',
    "max_depth": max_depth,
    "subsample": subsample, # collect 80% of the data only to prevent overfitting
    "colsample_bytree": colsample_bytree,
    "silent": 1,
    "seed": 0,
}

watchlist = [(dtrain, 'train')] # list of things to evaluate and print
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True) # find the best score
x_pred = np.expm1(gbm.predict(dtest))

elastic = ElasticNet(alpha=0.0005, l1_ratio=0.9)
elastic.fit(X_train, y)
elas_preds = np.expm1(elastic.predict(X_test))

lasso_preds = np.expm1(lasso_model.predict(X_test))
final_result=0.8*lasso_preds+0.2*x_pred
#+0.1*elas_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":final_result}, columns=['id', 'SalePrice'])
solution.to_csv("final_upload_11PM2.csv", index = False)


# In[ ]:

