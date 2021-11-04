import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import ward, fcluster
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from plotly.offline import plot


train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
train.drop(['Id'], axis=1, inplace=True)
for i in list(train):
    if type(train[i][0]) is not np.int64:
        train.drop(i,axis=1, inplace=True)
y = train['SalePrice']
train.drop(['SalePrice'], axis=1, inplace=True)

pca = PCA(17).fit_transform(train)
potentialRange = range(1,10)
kmeans = [KMeans(i) for i in potentialRange]
pcaScores = [kmeans[i].fit(pca).score(pca) for i in range(len(kmeans))]
processed = [go.Scatter(x=list(potentialRange),y=pcaScores)]
plot({'data':processed,'layout':{'title':'Elbow Method Results'}})

kmeans = KMeans(6)
kmeans.fit(pca)
tsne = TSNE(2).fit_transform(train)
fr = pd.DataFrame({'TSNE 1':tsne[:,0],'TSNE 2':tsne[:, 1],'clusterLabels':kmeans.labels_})
sns.lmplot(data=fr,x='TSNE 1',y='TSNE 2',hue='clusterLabels',fit_reg=False)

Normalizer().fit(train)
normalizedData = Normalizer().transform(train)
tsneRes = TSNE().fit_transform(normalizedData)
sns.scatterplot(x=tsneRes[:,0],y=tsneRes[:,1],hue=fcluster(ward(tsneRes),300, 'distance'),palette="Set1")
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.show()
