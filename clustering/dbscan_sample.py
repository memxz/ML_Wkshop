import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.cluster import DBSCAN


'''
Performs DBSCAN clustering.

"eps" is the radius of a dense region (with respect to point under consideration).
"min_samples" is the minimum number of points to form a dense region.
'''
def dbscan(X):
    algo = DBSCAN(eps=0.6, min_samples=4)
    
    # 'clusters' contain the cluster-assignment for each row of data
    clusters = algo.fit_predict(X)
    return clusters


'''
Plot cluster-assignments.
'''
def plot_cluster_assignments(X, clusters):
    sns.set(style='darkgrid')

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

    for i in np.unique(clusters):
        bool_arr = (clusters == i)

        label = 'Outlier' if i==-1 else 'Cluster ' + str(i+1)

        sns.scatterplot(x=X[bool_arr,0], y=X[bool_arr,1],
            label=label, ax=ax[0,0])
        sns.scatterplot(x=X[bool_arr,2], y=X[bool_arr,3],
            label=label, ax=ax[0,1])
        sns.scatterplot(x=X[bool_arr,0], y=X[bool_arr,2],
            label=label, ax=ax[1,0])
        sns.scatterplot(x=X[bool_arr,1], y=X[bool_arr,3],
            label=label, ax=ax[1,1])

    ax[0,0].set_xlabel('SepalLengthCm')
    ax[0,0].set_ylabel('SepalWidthCm')
    ax[0,0].set_title('Clustering (View 1)')

    ax[0,1].set_xlabel('PetalLengthCm')
    ax[0,1].set_ylabel('PetalWidthCm')
    ax[0,1].set_title('Clustering (View 2)')

    ax[1,0].set_xlabel('SepalLengthCm')
    ax[1,0].set_ylabel('PetalLengthCm')
    ax[1,0].set_title('Clustering (View 3)')

    ax[1,1].set_xlabel('SepalWidthCm')
    ax[1,1].set_ylabel('PetalWidthCm')
    ax[1,1].set_title('Clustering (View 4)')

    plt.tight_layout(pad=3)
    fig.suptitle(t='Clustering Results')
    plt.show()


'''
Plot cluster-assignments in 3D.
'''
def plot3d_cluster_assignments(X, clusters):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in np.unique(clusters):
        ax.scatter(xs=X[clusters==i,0], 
            ys=X[clusters==i,1],
            zs=X[clusters==i,2],
            label='Cluster ' + str(i+1))
            
    ax.set_xlabel('SepalLengthCm')		
    ax.set_ylabel('SepalWidthCm')		
    ax.set_zlabel('PetalLengthCm')

    ax.set_title('3D View of DBSCAN Clustering Results')
    plt.legend()
    plt.show()		





'''
Main Program
'''

# read dataset into a Pandas DataFrame
df = pd.read_csv('iris.csv')
print(df)

# getting rows of data between these two column-names
X = df.loc[:,'SepalLengthCm':'PetalWidthCm']
X = X.values   # converting Pandas DataFrame to NumPy array

# use DBSCAN to perform clustering
clusters = dbscan(X=X)
print(clusters)

# plot clustering results
plot_cluster_assignments(X=X, clusters=clusters)

# plot clustering results in 3D
plot3d_cluster_assignments(X=X, clusters=clusters)
