import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('train.csv')
col = list(df)
for i in col:
    if not type(df[i][0]) is np.int64:
        df = df.drop(i,axis=1)
price = df['SalePrice']
df = df.drop('SalePrice',axis=1)
df = df.drop('Id',axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(df)
dfx = pd.DataFrame(data=X,columns=df.columns[0:])
pca = PCA(n_components=None)
dfx_pca = pca.fit(dfx)
n_pca= dfx_pca.components_.shape[0]
most_important = [np.abs(dfx_pca.components_[i]).argmax() for i in range(17)]
pos_effect_idx = [dfx_pca.components_[i].argmax() for i in range(17)]
neg_effect_idx = [dfx_pca.components_[i].argmin() for i in range(17)]
pos_effect = [dfx_pca.components_[i][pos_effect_idx[i]] for i in range(17)]
neg_effect = [dfx_pca.components_[i][neg_effect_idx[i]] for i in range(17)]
pos_effect_feature = [list(df)[pos_effect_idx[i]] for i in range(17)]
neg_effect_feature = [list(df)[neg_effect_idx[i]] for i in range(17)]
print(pos_effect,pos_effect_feature)
print(neg_effect,neg_effect_feature)
most_important_features = []
for i in most_important:
    most_important_features.append(list(df)[i])
significant_components = list(dfx_pca.explained_variance_ratio_[0:17])

# res = []
# res.append([i for i in range(1,18)])
# res.append(significant_components)
# res.append(most_important_features)
# res = np.array(res).T
# head = ['PC-','explained_ratio','most_important_feature']
# pd.DataFrame(res).to_csv('pca_res.csv',header=head,index="Idx")

# plt.figure(figsize=(15,10))
# plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],
#             y=dfx_pca.explained_variance_ratio_,
#            s=200, alpha=0.75,c='orange',edgecolor='k')
# plt.grid(True)
# plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
# plt.xlabel("Principal components",fontsize=15)
# plt.xticks([i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],fontsize=15)
# plt.yticks(fontsize=15)
# plt.ylabel("Explained variance ratio",fontsize=15)
# plt.savefig("explained_pca.png")

# plt.figure()
# table = pd.read_csv('pca_res.csv')
# cell_text = []
# for row in range(17):
#     cell_text.append(table.iloc[row])
# plt.table(cellText=cell_text, colLabels=table.columns, loc='center')
# plt.axis('off')
# plt.savefig('pca_res.png')