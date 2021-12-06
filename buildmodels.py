from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse import hstack
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from collections import defaultdict

df = pd.read_csv('categorical.csv')
col = set(df.columns.values)
train_onehot = {}
vectorizer = CountVectorizer(lowercase=False, binary=True)
for i in col:
    train_onehot[i] = vectorizer.fit_transform(df[i].values.astype('U'))
train_sparse = hstack((train_onehot["RoofStyle"], train_onehot["HouseStyle"], train_onehot["BsmtFinType2"], train_onehot["KitchenQual"],train_onehot["Condition2"], train_onehot["BsmtQual"], train_onehot["SaleType"], train_onehot["LandSlope"], train_onehot["Street"], train_onehot["GarageFinish"], train_onehot["MasVnrType"], train_onehot["LotConfig"], train_onehot["HeatingQC"], train_onehot["SaleCondition"], train_onehot["BsmtCond"], train_onehot["Neighborhood"], train_onehot["BsmtExposure"], train_onehot["RoofMatl"], train_onehot["ExterCond"], train_onehot["Foundation"], train_onehot["LandContour"], train_onehot["Functional"], train_onehot["Electrical"], train_onehot["ExterQual"], train_onehot["MSZoning"], train_onehot["BsmtFinType1"], train_onehot["Exterior2nd"], train_onehot["CentralAir"], train_onehot["BldgType"], train_onehot["LotShape"], train_onehot["Heating"], train_onehot["GarageType"], train_onehot["Exterior1st"], train_onehot["GarageCond"], train_onehot["GarageQual"], train_onehot["Condition1"], train_onehot["PavedDrive"], train_onehot["Utilities"])).tocsr()

df_num = pd.read_csv('numerical.csv')
x_train = hstack((df_num.values,train_sparse)).tocsr()
y_train = pd.read_csv('train.csv')['SalePrice']
alpha = [1, 2, 3, 3.5, 4, 4.5, 5, 6, 7]
for i in tqdm(alpha):
    model = Ridge(solver="auto", random_state=42, alpha=i)
    model.fit(x_train, y_train)
ridge_preds_tr = model.predict(x_train)
print(ridge_preds_tr)
fselect = SelectKBest(f_regression, k=200)
train_features = fselect.fit_transform(train_sparse, y_train)
x_train = hstack((df_num.values, ridge_preds_tr.reshape(-1,1), train_features)).tocsr()