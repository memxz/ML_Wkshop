import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


'''
Generate a Dendrogram to visual the dataset for Hierarchical Clustering.

For the terms "linkage" and "ward", refer to this article - https://towardsdatascience.com/introduction-to-hierarchical-clustering-part-1-theory-linkage-and-affinity-e3b6a4817702 
'''
def make_dendrogram(X):
    plt.figure(figsize=(25, 10))

    dendrogram(
        linkage(X, 'ward'),  # generate the linkage matrix
        leaf_font_size=8     # font size for the x axis labels
    )

    plt.title('Iris Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')

    # notice that there are 3 clusters (intersections) at a distance of 8 units
    plt.axhline(y=8)    
    plt.show()


'''
Generate cluster-assignments using Hierarchical Clustering.
'''
def kmeans_clustering(X):
    algo = AgglomerativeClustering(linkage="ward", n_clusters=3)
    return algo.fit_predict(X)


'''
Plot the cluster-assignments.
'''
def plot_cluster_assignments(X):
    _, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

    for i in np.unique(clusters):
        # returns an array of booleans (values that are either 'true' or 'false') 
        # here, elements in clusters that are of value i will be 'true', while the 
        # rest will be 'false' 
        # for example, bool_arr[0] contains true only if clusters[0] has value i,
        # else, bool_arr[0] contains false
        bool_arr = (clusters == i)

        # X[bool_arr] returns rows whose bool_arr[row-nos] contains true
        sns.scatterplot(x=X[bool_arr,0], y=X[bool_arr,1],
            label='Cluster ' + str(i+1), ax=ax[0,0])
        sns.scatterplot(x=X[bool_arr,2], y=X[bool_arr,3],
            label='Cluster ' + str(i+1), ax=ax[0,1])
        sns.scatterplot(x=X[bool_arr,0], y=X[bool_arr,2],
            label='Cluster ' + str(i+1), ax=ax[1,0])
        sns.scatterplot(x=X[bool_arr,1], y=X[bool_arr,3],
            label='Cluster ' + str(i+1), ax=ax[1,1])

    ax[0,0].set_xlabel('SepalLengthCm')
    ax[0,0].set_ylabel('SepalWidthCm')
    ax[0,0].set_title('Iris Clustering (plot 1)')

    ax[0,1].set_xlabel('PetalLengthCm')
    ax[0,1].set_ylabel('PetalWidthCm')
    ax[0,1].set_title('Iris Clustering (plot 2)')

    ax[1,0].set_xlabel('SepalLengthCm')
    ax[1,0].set_ylabel('PetalLengthCm')
    ax[1,0].set_title('Iris Clustering (plot 3)')

    ax[1,1].set_xlabel('SepalWidthCm')
    ax[1,1].set_ylabel('PetalWidthCm')
    ax[1,1].set_title('Iris Clustering (plot 4)')

    plt.tight_layout(pad=3)
    plt.show()



'''
Main Program
'''

#Read dataset into a Pandas DataFrame
df = pd.read_csv('iris.csv')
print(df)

# remove first and last columns
X = df.iloc[:, 1:-1].values    
make_dendrogram(X)

# perform clustering using k-means algo
clusters = kmeans_clustering(X)
print(clusters)

# plot the assigned clusterings 
plot_cluster_assignments(X)
