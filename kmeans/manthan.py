#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv


# In[4]:


dataset = pd.read_csv('RMS_Crime_Incidents_2.csv')
X = dataset.iloc[:, [0, 1]].values


# In[5]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 9):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 9), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[6]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[7]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Crime mapping')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.legend()
plt.show()


# In[8]:


df = pd.read_csv('RMS_Crime_Incidents_2.csv')
df.head(5)


# In[9]:


X=df[['oid','latitude','longitude']]
X.head(5)


# In[10]:


dataset.head(5)


# In[11]:


kmeans = KMeans(n_clusters = 3, init ='k-means++')
kmeans.fit(X[X.columns[1:3]]) # Compute k-means clustering.
X['cluster_label'] = kmeans.fit_predict(X[X.columns[1:3]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(X[X.columns[1:3]]) # Labels of each point


# In[12]:


X = X[['oid','cluster_label']]
X.head(10)


# In[13]:


clustered_data = df.merge(X, left_on='oid', right_on='oid')
clustered_data.head(5)


# In[14]:


clustered_data.to_csv ('clustereddata2.csv', index=None, header = True)


# In[15]:


centers = kmeans.cluster_centers_
print(centers)


# In[21]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(clustered_data['cluster_label'],kmeans.labels_))
print(classification_report(clustered_data['cluster_label'],kmeans.labels_))


# In[ ]:




