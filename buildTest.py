from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy.sparse import hstack
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression

df = pd.read_csv('categorical_test.csv')
col = set(df.columns.values)
train_onehot = {}
vectorizer = CountVectorizer(lowercase=False, binary=True)
for i in col:
    train_onehot[i] = vectorizer.fit_transform(df[i].values.astype('U'))
train_sparse = hstack((train_onehot["RoofStyle"], train_onehot["HouseStyle"], train_onehot["BsmtFinType2"], train_onehot["KitchenQual"],train_onehot["Condition2"], train_onehot["BsmtQual"], train_onehot["SaleType"], train_onehot["LandSlope"], train_onehot["Street"], train_onehot["GarageFinish"], train_onehot["MasVnrType"], train_onehot["LotConfig"], train_onehot["HeatingQC"], train_onehot["SaleCondition"], train_onehot["BsmtCond"], train_onehot["Neighborhood"], train_onehot["BsmtExposure"], train_onehot["RoofMatl"], train_onehot["ExterCond"], train_onehot["Foundation"], train_onehot["LandContour"], train_onehot["Functional"], train_onehot["Electrical"], train_onehot["ExterQual"], train_onehot["MSZoning"], train_onehot["BsmtFinType1"], train_onehot["Exterior2nd"], train_onehot["CentralAir"], train_onehot["BldgType"], train_onehot["LotShape"], train_onehot["Heating"], train_onehot["GarageType"], train_onehot["Exterior1st"], train_onehot["GarageCond"], train_onehot["GarageQual"], train_onehot["Condition1"], train_onehot["PavedDrive"], train_onehot["Utilities"])).tocsr()
df_num = pd.read_csv('numerical_test.csv')
x_test = hstack((df_num.values,train_sparse)).tocsr()