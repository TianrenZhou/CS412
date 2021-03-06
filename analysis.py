from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse import hstack
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from matplotlib.pyplot import figure


df = pd.read_csv('categorical.csv')
df_test = pd.read_csv('categorical_test.csv')
col = set(df.columns.values)
train_onehot = {}
train_onehot_test = {}
vectorizer = CountVectorizer(lowercase=False, binary=True)
for i in col:
    train_onehot[i] = vectorizer.fit_transform(df[i].values.astype('U'))
    train_onehot_test[i] = vectorizer.transform(df_test[i].values.astype('U'))
train_sparse = hstack((train_onehot["RoofStyle"], train_onehot["HouseStyle"], train_onehot["BsmtFinType2"], train_onehot["KitchenQual"],train_onehot["Condition2"], train_onehot["BsmtQual"], train_onehot["SaleType"], train_onehot["LandSlope"], train_onehot["Street"], train_onehot["GarageFinish"], train_onehot["MasVnrType"], train_onehot["LotConfig"], train_onehot["HeatingQC"], train_onehot["SaleCondition"], train_onehot["BsmtCond"], train_onehot["Neighborhood"], train_onehot["BsmtExposure"], train_onehot["RoofMatl"], train_onehot["ExterCond"], train_onehot["Foundation"], train_onehot["LandContour"], train_onehot["Functional"], train_onehot["Electrical"], train_onehot["ExterQual"], train_onehot["MSZoning"], train_onehot["BsmtFinType1"], train_onehot["Exterior2nd"], train_onehot["CentralAir"], train_onehot["BldgType"], train_onehot["LotShape"], train_onehot["Heating"], train_onehot["GarageType"], train_onehot["Exterior1st"], train_onehot["GarageCond"], train_onehot["GarageQual"], train_onehot["Condition1"], train_onehot["PavedDrive"], train_onehot["Utilities"])).tocsr()
train_sparse_test = hstack((train_onehot_test["RoofStyle"], train_onehot_test["HouseStyle"], train_onehot_test["BsmtFinType2"], train_onehot_test["KitchenQual"],train_onehot_test["Condition2"], train_onehot_test["BsmtQual"], train_onehot_test["SaleType"], train_onehot_test["LandSlope"], train_onehot_test["Street"], train_onehot_test["GarageFinish"], train_onehot_test["MasVnrType"], train_onehot_test["LotConfig"], train_onehot_test["HeatingQC"], train_onehot_test["SaleCondition"], train_onehot_test["BsmtCond"], train_onehot_test["Neighborhood"], train_onehot_test["BsmtExposure"], train_onehot_test["RoofMatl"], train_onehot_test["ExterCond"], train_onehot_test["Foundation"], train_onehot_test["LandContour"], train_onehot_test["Functional"], train_onehot_test["Electrical"], train_onehot_test["ExterQual"], train_onehot_test["MSZoning"], train_onehot_test["BsmtFinType1"], train_onehot_test["Exterior2nd"], train_onehot_test["CentralAir"], train_onehot_test["BldgType"], train_onehot_test["LotShape"], train_onehot_test["Heating"], train_onehot_test["GarageType"], train_onehot_test["Exterior1st"], train_onehot_test["GarageCond"], train_onehot_test["GarageQual"], train_onehot_test["Condition1"], train_onehot_test["PavedDrive"], train_onehot_test["Utilities"])).tocsr()
df_num = pd.read_csv('numerical.csv')
df_num_test = pd.read_csv('numerical_test.csv')
x_train_pre = hstack((df_num.values,train_sparse)).tocsr()
y_train_pre = np.log1p(pd.read_csv('train.csv')['SalePrice'])
#x_train_pre.drop(['price'], axis=1, inplace=True)
x_train, x_cv, y_train, y_cv = train_test_split(x_train_pre, y_train_pre, test_size=0.1, random_state=66)
#x_test = hstack((df_num_test.values,train_sparse_test)).tocsr()
#print('Train size: {}, CV size: {}, test size: {}' .format(x_train.shape, x_cv.shape, x_test.shape))

# first ridge fitting
alpha = list(np.arange(0, 10, 0.5))
cv_rmsle = []
for i in tqdm(alpha):
    model = Ridge(solver="auto", random_state=66, alpha=i)
    model.fit(x_train, y_train)
    y_cv_preds = model.predict(x_cv)
    cv_rmsle.append(np.sqrt(mean_squared_error(y_cv, y_cv_preds, squared=True)))
for i in range(len(cv_rmsle)):
    print('RMSLE for alpha = ', alpha[i], 'is', cv_rmsle[i])
best_alpha = np.argmin(cv_rmsle)
fig, ax = plt.subplots()
ax.plot(alpha, cv_rmsle)
ax.scatter(alpha, cv_rmsle)
for i, txt in enumerate(np.round(cv_rmsle,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_rmsle[i]))
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.show()

# second ridge fitting
alpha = list(np.arange(0, 1, 0.1))
cv_rmsle = []
for i in tqdm(alpha):
    model = Ridge(solver="auto", random_state=66, alpha=i)
    model.fit(x_train, y_train)
    y_cv_preds = model.predict(x_cv)
    cv_rmsle.append(np.sqrt(mean_squared_error(y_cv, y_cv_preds, squared=True)))
for i in range(len(cv_rmsle)):
    print('RMSLE for alpha = ', alpha[i], 'is', cv_rmsle[i])
best_alpha = np.argmin(cv_rmsle)
fig, ax = plt.subplots()
ax.plot(alpha, cv_rmsle)
ax.scatter(alpha, cv_rmsle)
for i, txt in enumerate(np.round(cv_rmsle,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_rmsle[i]))
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.show()

# ridge train and cv results
print("Best alpha: ",  alpha[best_alpha])
model = Ridge(solver="auto", random_state=66, alpha=alpha[best_alpha])
model.fit(x_train, y_train)
ridge_preds_train = model.predict(x_train)
ridge_preds_cv = model.predict(x_cv)
print('Train RMSLE:', np.sqrt(mean_squared_error(y_train, ridge_preds_train, squared=True)))
ridge_rmsle = np.sqrt(mean_squared_error(y_cv, ridge_preds_cv, squared=True))
print("Cross validation RMSLE: ", ridge_rmsle)

# first SVM
c = list(np.arange(1, 10, 0.5))
cv_rmsle = []
for i in tqdm(c):
    model = SVR(C=i)
    model.fit(x_train, y_train)
    y_cv_preds = model.predict(x_cv)
    cv_rmsle.append(np.sqrt(mean_squared_error(y_cv, y_cv_preds, squared=True)))
for i in range(len(cv_rmsle)):
    print('RMSLE for c = ', c[i], 'is', cv_rmsle[i])
best_c = np.argmin(cv_rmsle)
fig, ax = plt.subplots()
ax.plot(c, cv_rmsle)
ax.scatter(c, cv_rmsle)
for i, txt in enumerate(np.round(cv_rmsle,3)):
    ax.annotate((c[i],np.round(txt,3)), (c[i],cv_rmsle[i]))
plt.title("Cross Validation Error for each c")
plt.xlabel("C")
plt.ylabel("Error")
plt.savefig('Best_C1.png')


# second SVM
c = list(np.arange(1, 2, 0.1))
cv_rmsle = []
for i in tqdm(c):
    model = SVR(C=i)
    model.fit(x_train, y_train)
    y_cv_preds = model.predict(x_cv)
    cv_rmsle.append(np.sqrt(mean_squared_error(y_cv, y_cv_preds, squared=True)))
for i in range(len(cv_rmsle)):
    print('RMSLE for c = ', c[i], 'is', cv_rmsle[i])
best_c = np.argmin(cv_rmsle)
fig, ax = plt.subplots()
ax.plot(c, cv_rmsle)
ax.scatter(c, cv_rmsle)
for i, txt in enumerate(np.round(cv_rmsle,3)):
    ax.annotate((c[i],np.round(txt,3)), (c[i],cv_rmsle[i]))
plt.title("Cross Validation Error for each c")
plt.xlabel("C")
plt.ylabel("Error")
plt.savefig('Best_C2.png')


# SVM train and cv results
print("Best c: ",  c[best_c])
model = SVR(C=c[best_c])
model.fit(x_train, y_train)
svm_preds_train = model.predict(x_train)
svm_preds_cv = model.predict(x_cv)
print('SVM Train RMSLE:', np.sqrt(mean_squared_error(y_train, svm_preds_train, squared=True)))
svm_rmsle = np.sqrt(mean_squared_error(y_cv, svm_preds_cv, squared=True))
print("SVM Cross validation RMSLE: ", svm_rmsle)


# random forest train and cv results
# model = RandomForestRegressor(max_depth = 50, random_state=66)
# model.fit(x_train, y_train)
# rf_preds_train = model.predict(x_train)
# rf_preds_cv = model.predict(x_cv)
# print('Train RMSLE:', np.sqrt(mean_squared_error(y_train, rf_preds_train, squared=True)))
# rf_rmsle = np.sqrt(mean_squared_error(y_cv, rf_preds_cv, squared=True))
# print("Cross validation RMSLE: ", rf_rmsle)


# # lightGBM train
# lgb_model = LGBMRegressor()
# params = {'learning_rate': uniform(0, 1),
#           'n_estimators': sp_randint(200, 1500),
#           'num_leaves': sp_randint(20, 200),
#           'max_depth': sp_randint(2, 15),
#           'min_child_weight': uniform(0, 2),
#           'colsample_bytree': uniform(0, 1),
#          }
# lgb_random = RandomizedSearchCV(lgb_model, param_distributions=params, n_iter=10, cv=3, random_state=66, 
#                                 scoring='neg_root_mean_squared_error', verbose=10, return_train_score=True)
# lgb_random = lgb_random.fit(x_train, y_train)
# best_params = lgb_random.best_params_
# print('best params are:', best_params)


# # lightGBM cv
# model = LGBMRegressor(**best_params, random_state=66, n_jobs=-1)
# model.fit(x_train, y_train)
# lgb_preds_tr = model.predict(x_train)
# lgb_preds_cv = model.predict(x_cv)
# print('Train RMSLE:', np.sqrt(mean_squared_error(y_train, lgb_preds_tr, squared=True)))
# lgb_rmsle = np.sqrt(mean_squared_error(y_cv, lgb_preds_cv, squared=True))
# print("Cross validation RMSLE: ", lgb_rmsle)


# # build test

# # print("Best alpha: ",  alpha[best_alpha])
# # model = Ridge(solver="auto", random_state=66, alpha=alpha[best_alpha])
# # model.fit(x_train, y_train)
# # model2 = Ridge(solver="auto", random_state=66, alpha=alpha[best_alpha])
# # model2.fit(x_train_pre, y_train_pre)
# # ridge_preds_train = model.predict(x_train)
# # ridge_preds_train2 = model2.predict(x_train)
# # print('Train RMSLE:', np.sqrt(mean_squared_error(y_train, ridge_preds_train, squared=True)))
# # ridge_rmsle = np.sqrt(mean_squared_error(y_train, ridge_preds_train2, squared=True))
# # print("Cross validation RMSLE: ", ridge_rmsle)


# ids = np.array(list(range(1461, 2920)))

# model = Ridge(solver="auto", random_state=66, alpha=alpha[best_alpha])
# model.fit(x_train_pre, y_train_pre)
# ridge_preds_test = np.expm1(model.predict(x_test))
# ridge_preds_test = np.stack((ids, ridge_preds_test), axis=1)
# np.savetxt('ridge_result.csv', ridge_preds_test, delimiter=",", fmt='%i, %f', header="Id,SalePrice", comments='')
# #print('Ridge Test RMSLE:', np.sqrt(mean_squared_error(y_test, ridge_preds_test, squared=True)))


# model = SVR(C=c[best_c])
# model.fit(x_train_pre, y_train_pre)
# svm_preds_test= np.expm1(model.predict(x_test))
# svm_preds_test = np.stack((ids, svm_preds_test), axis=1)
# np.savetxt('svm_preds_test.csv', svm_preds_test, delimiter=",", fmt='%i, %f', header="Id,SalePrice", comments='')
# #print('SVM Test RMSLE:', np.sqrt(mean_squared_error(y_test, svm_preds_test, squared=True)))

# model = RandomForestRegressor(max_depth = 50, random_state=66)
# model.fit(x_train_pre, y_train_pre)
# rf_preds_test= np.expm1(model.predict(x_test))
# rf_preds_test = np.stack((ids, rf_preds_test), axis=1)
# np.savetxt('rf_preds_test.csv', rf_preds_test, delimiter=",", fmt='%i, %f', header="Id,SalePrice", comments='')
# #print('Random Forest Test RMSLE:', np.sqrt(mean_squared_error(y_test, rf_preds_test, squared=True)))

# model = LGBMRegressor(**best_params, random_state=66, n_jobs=-1)
# model.fit(x_train_pre, y_train_pre)
# lgb_preds_test= np.expm1(model.predict(x_test))
# lgb_preds_test = np.stack((ids, lgb_preds_test), axis=1)
# np.savetxt('lgb_preds_test.csv', lgb_preds_test, delimiter=",", fmt='%i, %f', header="Id,SalePrice", comments='')
# #print('LGB Test RMSLE:', np.sqrt(mean_squared_error(y_test, lgb_preds_test, squared=True)))
