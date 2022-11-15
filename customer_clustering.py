import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Data/segmentation data.csv", index_col="ID")

data.head()

data.isna().sum().sum()

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# initalise kmean cluster
cluster = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300
)

cluster.fit(scaled_data)
# Lowest SSE value

cluster.inertia_

# Final locations of the centroid

cluster.cluster_centers_

cluster.n_iter_

# how to determine a appropriate amount of clusters

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300, }

# SSE (sum of squared error) used to messures the error of cluster
# silhouette coefficient is a mesure of cluster cohesion and sperations
# the more to 1 it is the closer the clusters are to other clusters
sse = []
silhouette_coefficient = []
for k in range(2, 26):
    cluster = KMeans(n_clusters=k, **kmeans_kwargs)
    cluster.fit(scaled_data)
    sse.append(cluster.inertia_)
    score = silhouette_score(scaled_data, cluster.labels_)
    silhouette_coefficient.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 26), sse)
plt.xticks(range(1, 26))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

plt.style.use("fivethirtyeight")
plt.plot(range(2, 26), silhouette_coefficient)
plt.xticks(range(2, 26))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


## Got the ideal amount of clusters and wil now plot the cluster

cluster = KMeans(
    init="random",
    n_clusters=5,
    n_init=10,
    max_iter=300
)

cluster.fit(scaled_data)

pred = cluster.fit_predict(scaled_data)
pred

#Add the cluster labels to original data
data["cluster"] = cluster.labels_

data["cluster"].value_counts()

plt.figure(figsize=(10,7))
plt.scatter(data['Age'],data['Income'],c=cluster.labels_,cmap='rainbow')
