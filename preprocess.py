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


#preprogress missing value 


# In[4]:


missing_val= pd.DataFrame(df_train.isnull().sum()[df_train.isnull().sum()!=0]                          .sort_values(ascending = False)).rename(columns = {0:'num_miss'})
missing_val['missing_perc'] = (missing_val/df_train.shape[0]*100).round(1)
missing_val = missing_val.query('missing_perc > 20')


# In[5]:


missing_val


# In[6]:


drop_cols = missing_val.index.to_list()
drop_cols
df_train.drop(['Id'],axis=1,inplace=True)
df_train.drop(columns=drop_cols,axis=1,inplace=True)


# In[7]:


numerical_features = df_train.select_dtypes(exclude='object')
categorical_features = df_train.select_dtypes(include='object')


# In[8]:


categorical_features.describe()
categorical_features.shape


# In[10]:


categorical_missing = categorical_features.columns[categorical_features.isnull().any()]


# In[11]:


categorical_missing


# In[13]:


len(categorical_features['CentralAir'])


# In[14]:


for i in range (len(categorical_features['CentralAir'])):
    if(categorical_features['CentralAir'][i] == 'Y'):
        categorical_features['CentralAir'][i] = 'Yes'
    if(categorical_features['CentralAir'][i] == 'N'):
        categorical_features['CentralAir'][i] = 'No'


# In[15]:


for i in range (len(categorical_features['PavedDrive'])):
    if(categorical_features['PavedDrive'][i] == 'Y'):
        categorical_features['PavedDrive'][i] = 'Paved'
    if(categorical_features['PavedDrive'][i] == 'N'):
        categorical_features['PavedDrive'][i] = 'Dirt'
    if(categorical_features['PavedDrive'][i] == 'P'):
        categorical_features['PavedDrive'][i] = 'Partial'


# In[16]:


categorical_features.to_csv('categorical.csv',index =False)


# In[17]:


categorical_features.describe()


# In[ ]:


#numerical


# In[20]:


numerical_features = df_train.select_dtypes(exclude='object')
numerical_features.shape


# In[21]:


numerical_features.isna().sum()


# In[22]:


numerical_features['LotFrontage'].describe()


# In[23]:


numerical_features['LotFrontage'].fillna(np.random.randint(59,80), inplace = True)
numerical_features['LotFrontage'].isna().sum()


# In[24]:


numerical_features['GarageYrBlt'].describe()


# In[25]:


numerical_features['GarageYrBlt'].fillna(np.random.randint(1961,2002), inplace = True)
numerical_features['GarageYrBlt'].isna().sum()


# In[26]:



numerical_features['MasVnrArea'].fillna(0, inplace = True)
numerical_features['MasVnrArea'].isna().sum()


# In[ ]:


# Normalization on numerical features


# In[30]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[32]:


numerical_features = normalize(numerical_features)


# In[35]:


numerical_features.drop(['SalePrice'],axis=1,inplace=True)


# In[37]:


numerical_features.to_csv('numerical.csv',index =False)


# In[ ]:





# In[47]:


# test
df_test=pd.read_csv("test.csv")

numerical_test = df_test.select_dtypes(exclude='object')
categorical_test = df_test.select_dtypes(include='object')

for col in numerical_test:
    if col not in numerical_features:
        numerical_test.drop(col, axis = 1, inplace = True)
for col in categorical_test:
    if col not in categorical_features:
        categorical_test.drop(col, axis = 1, inplace = True)


# In[48]:


numerical_test.isna().sum()


# In[49]:


numerical_test['LotFrontage'].describe()


# In[50]:


numerical_test['LotFrontage'].fillna(np.random.randint(59,80), inplace = True)
numerical_test['LotFrontage'].isna().sum()


# In[51]:


numerical_test['GarageYrBlt'].fillna(np.random.randint(1961,2002), inplace = True)
numerical_test['GarageYrBlt'].isna().sum()


# In[52]:



numerical_test['MasVnrArea'].fillna(0, inplace = True)
numerical_test['MasVnrArea'].isna().sum()


# In[53]:


numerical_test.to_csv('numerical_test.csv',index =False)


# In[57]:


for i in range (len(categorical_test['CentralAir'])):
    if(categorical_test['CentralAir'][i] == 'Y'):
        categorical_test['CentralAir'][i] = 'Yes'
    if(categorical_test['CentralAir'][i] == 'N'):
        categorical_test['CentralAir'][i] = 'No'


# In[58]:


for i in range (len(categorical_test['PavedDrive'])):
    if(categorical_test['PavedDrive'][i] == 'Y'):
        categorical_test['PavedDrive'][i] = 'Paved'
    if(categorical_test['PavedDrive'][i] == 'N'):
        categorical_test['PavedDrive'][i] = 'Dirt'
    if(categorical_test['PavedDrive'][i] == 'P'):
        categorical_test['PavedDrive'][i] = 'Partial'


# In[59]:


categorical_test.to_csv('categorical_test.csv',index =False)


# In[ ]:





# In[ ]:





# In[ ]:




