#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from mpl_toolkits import mplot3d
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)


# In[2]:


df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")


# In[3]:


# process missing value
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.figure(figsize=(10,8))
sns.heatmap(df_train.isnull(), cbar = False, cmap="gray")


# In[5]:


missing_val= pd.DataFrame(df_train.isnull().sum()[df_train.isnull().sum()!=0]                          .sort_values(ascending = False)).rename(columns = {0:'num_miss'})
missing_val['missing_perc'] = (missing_val/df_train.shape[0]*100).round(1)
missing_val = missing_val.query('missing_perc > 20')


# In[6]:


missing_val


# In[7]:


#we need to drop the attributes missing_perc > 20
drop_cols = missing_val.index.to_list()
drop_cols
df_train.drop(['Id'],axis=1,inplace=True)
df_train.drop(columns=drop_cols,axis=1,inplace=True)


# In[8]:


numerical_features = df_train.select_dtypes(exclude='object')
categorical_features = df_train.select_dtypes(include='object')
numerical_features.describe()


# In[9]:


categorical_features.describe()


# In[10]:


# correlation between 'SalePrice' and other columns of the training data set 


# In[11]:


numerical_corr = numerical_features.corr()['SalePrice'][:-1]
plt.figure(figsize=(15,10))
numerical_corr


# In[12]:


numerical_best = numerical_corr[abs(numerical_corr) > 0.25].sort_values(ascending=False)


# In[13]:


for feature in numerical_best.index:
    numerical_corr.drop(feature,inplace = True)
for feature in numerical_corr.index:
    df_train.drop(feature,axis = 1,inplace = True)
    numerical_features.drop(feature,axis = 1,inplace = True)


# In[14]:


df_train.shape


# In[15]:


numerical_features.isna().sum()


# In[16]:


numerical_features['LotFrontage'].hist(bins = 40)


# In[17]:


numerical_features['LotFrontage'].describe()


# In[18]:


df_train['LotFrontage'].fillna(np.random.randint(59,80), inplace = True)
df_train['LotFrontage'].isna().sum()


# In[19]:


numerical_features['GarageYrBlt'].hist(bins = 40)


# In[20]:


numerical_features['GarageYrBlt'].describe()


# In[21]:


df_train['GarageYrBlt'].fillna(np.random.randint(1961,2002), inplace = True)
df_train['GarageYrBlt'].isna().sum()


# In[22]:


numerical_features.MasVnrArea.fillna(0, inplace = True)
df_train['MasVnrArea'].fillna(0, inplace = True)
df_train['MasVnrArea'].isna().sum()


# In[23]:


df_train.shape


# In[24]:


sns.scatterplot(data=df_train, x='OverallQual', y='SalePrice')
plt.axvline(x=9.5, color='red', ls='--')
plt.axhline(y=200000, color='red', ls='--')
plt.axhline(y=650000, color='red', ls='--')


# In[25]:


outlier_table_1 = df_train[(df_train['OverallQual']>9.5) & (df_train['SalePrice']<200000)][['SalePrice', 'OverallQual']]
outlier_table_2 = df_train[(df_train['SalePrice']>650000)][['SalePrice', 'OverallQual']]
pd.concat([outlier_table_1, outlier_table_2], axis=0)


# In[26]:


sns.scatterplot(data=df_train, x='GrLivArea', y='SalePrice')
plt.axvline(x=4000, color='red', ls='--')
df_train[(df_train['GrLivArea']>4000)][['SalePrice', 'GrLivArea']]


# In[27]:


sns.scatterplot(data=df_train, x='GarageCars', y='SalePrice')
plt.axhline(y=650000, color='red', ls='--')
df_train[df_train['SalePrice']>650000][['SalePrice', 'GarageCars']]


# In[28]:


sns.scatterplot(data=df_train, x='GarageArea', y='SalePrice')
plt.axhline(y=650000, color='red', ls='--')
plt.axhline(y=300000, color='red', ls='--')
plt.axvline(x=1200, color='red', ls='--')
outlier_table_1 = df_train[df_train['SalePrice']>650000][['SalePrice', 'GarageArea']]
outlier_table_2 = df_train[(df_train['GarageArea']>1200) & (df_train['SalePrice']<300000)][['SalePrice', 'GarageArea']]
pd.concat([outlier_table_1, outlier_table_2], axis=0)


# In[29]:


index_drop = df_train[(df_train['GrLivArea']>4000)].index
df_train = df_train.drop(index_drop, axis=0)
index_drop = df_train[(df_train['GarageArea']>1200) & (df_train['SalePrice']<300000)].index
df_train = df_train.drop(index_drop, axis=0)
df_train.corr()['SalePrice'].sort_values()


# In[30]:


df_train.shape


# In[31]:


df_train.SalePrice.describe()


# In[ ]:


#Check Normality assumption of the SalePrice


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sigma = df_train.SalePrice.std()
mu = df_train.SalePrice.mean()
med = df_train.SalePrice.median()
mode = df_train.SalePrice.mode().to_numpy()
plt.figure(figsize=(12,8))
plt.title(f'Untransformed Data, Skew: {stats.skew(df_train.SalePrice):.3f}')
sns.distplot(df_train.SalePrice)
plt.axvline(mode, linestyle='--', color='green', label='mode')
plt.axvline(med, linestyle='--', color='blue', label='median')
plt.axvline(mu, linestyle='--', color='red', label='mean')
plt.legend()


# In[43]:


plt.figure(figsize=(12,8))
plt.title(f'Log transform, Skew: {stats.skew(np.log(df_train.SalePrice)):.3f}')
sns.distplot(np.log(df_train['SalePrice']))


# In[32]:


categorical_missing = categorical_features.columns[categorical_features.isnull().any()]
categorical_missing


# In[33]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder


# In[34]:


imputer = SimpleImputer(missing_values = np.NaN,strategy = 'most_frequent')
for feature in categorical_missing:
     categorical_features[feature] = imputer.fit_transform(categorical_features[feature].values.reshape(-1,1))
     df_train[feature] = imputer.fit_transform(df_train[feature].values.reshape(-1,1))


# In[35]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in categorical_features.columns:
    categorical_features[feature]=le.fit_transform(categorical_features[feature])
    df_train[feature]=le.fit_transform(df_train[feature])


# In[36]:


categorical_features.head()


# In[37]:


# linear model
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
# x are the indepedant/ explanatory/ regressor variables
# y is the depedant/ explained/ regressed variable
x = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[39]:


# creating a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_predicted = linear_model.predict(X_test)


# In[43]:


# finding the score, mean absolute error, mean squared error and the root mean squared error
score = linear_model.score(X_train, y_train)
MAE = metrics.mean_absolute_error(y_test, y_predicted)
MSE = metrics.mean_squared_error(y_test, y_predicted)
RMSE = np.sqrt(MSE)

linear = pd.DataFrame([score, MAE, MSE, RMSE], index=['score', 'MAE', 'MSE', 'RMSE'], columns=['Linear'])
linear


# In[50]:


for col in df_test.columns:
    if col not in df_train.columns:
        df_test.drop(col, axis = 1, inplace = True)


# In[53]:


num_test = df_test.select_dtypes(include=['number'])
cat_test = df_test.select_dtypes(include=['object'])
num_test.isna().sum()


# In[70]:


df_test['LotFrontage'].fillna(np.random.randint(58,80), inplace = True)
df_test['MasVnrArea'].fillna(df_test.MasVnrArea.median(), inplace = True)
df_test['BsmtFinSF1'].fillna(df_test.BsmtFinSF1.median(), inplace = True)
df_test['TotalBsmtSF'].fillna(df_test.TotalBsmtSF.median(), inplace = True)
df_test['GarageYrBlt'].fillna(np.random.randint(1959,2002), inplace = True)
df_test['GarageArea'].fillna(df_test.GarageArea.median(), inplace = True)
df_test['GarageCars'].fillna(df_test.GarageCars.median(), inplace = True)


# In[71]:


cat_missing_test = cat_test.columns[cat_test.isnull().any()]
imputer = SimpleImputer(missing_values = np.NaN,strategy = 'most_frequent')
for feature in cat_missing_test:
     cat_test[feature] = imputer.fit_transform(cat_test[feature].values.reshape(-1,1))
     df_test[feature] = imputer.fit_transform(df_test[feature].values.reshape(-1,1))
for feature in cat_test.columns:
    cat_test[feature]=le.fit_transform(cat_test[feature])
    df_test[feature]=le.fit_transform(df_test[feature])


# In[ ]:


pred_y = linear_model.predict(df_test)
pred_y


# In[85]:


sample = pd.DataFrame()
sample['Id'] = range(1461,2920)
sample['SalePrice'] = pred_y


# In[87]:


sample.to_csv('my_submission.csv',index =False)


# In[ ]:




