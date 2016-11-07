from scaling_features import *

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


#LASSO MODEL
clf1 = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 5e-4])
clf1.fit(X_train, y)
lasso_preds = np.expm1(clf1.predict(X_test))

#ELASTIC NET
clf2 = ElasticNet(alpha=0.0005, l1_ratio=0.9)
clf2.fit(X_train, y)
elas_preds = np.expm1(clf2.predict(X_test))

#XGBOOST
clf3=xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0.045,                 
                 learning_rate=0.07,
                 max_depth=20,
                 min_child_weight=1.5,
                 n_estimators=300,                                                                    
                 reg_alpha=0.65,
                 reg_lambda=0.45,
                 subsample=0.95)

clf3.fit(X_train, y)
xgb_preds = np.expm1(clf3.predict(X_test))

final_result = 0.50*lasso_preds + 0.24*xgb_preds+0.26*elas_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":final_result}, columns=['id', 'SalePrice'])
solution.to_csv("Persistence_submission.csv", index = False)
