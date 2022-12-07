import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans


'''
Display the distribution for each attribute in our dataset.
'''
def plot_dist_by_attr(df):
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    sns.histplot(data=df['SepalLengthCm'], color='coral', kde=True, ax=ax[0,0])
    sns.histplot(data=df['SepalWidthCm'], color='cornflowerblue', kde=True, ax=ax[0,1])
    sns.histplot(data=df['PetalLengthCm'], color='violet', kde=True, ax=ax[1,0])
    sns.histplot(data=df['PetalWidthCm'], color='limegreen', kde=True, ax=ax[1,1])

    fig.suptitle(t="Distribution Of Attributes")
    plt.show()


'''
Display the KDE plots for each attribute by Species.

A Kernel Density Estimate (KDE) plot is a method for visualizing the 
distribution of observations in a dataset, analogous to a histogram. 
A KDE plot represents the data using a continuous probability density curve.
'''
def kde_plot(df):
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))

    sns.kdeplot(data=df, x='SepalLengthCm', hue='Species', ax=ax[0,0])
    sns.kdeplot(data=df, x='SepalWidthCm', hue='Species', ax=ax[0,1])
    sns.kdeplot(data=df, x='PetalLengthCm', hue='Species', ax=ax[1,0])
    sns.kdeplot(data=df, x='PetalWidthCm', hue='Species', ax=ax[1,1])    

    fig.suptitle(t="Distribution of Attributes by Species")
    plt.show()


'''
Plot Clustering Results.
'''
def plot_cluster_results(X, clusters):
    sns.set(style='darkgrid')
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,10))

    for i in np.unique(clusters):
        bool_arr = (clusters == i)
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
Plot Within-Cluster Sum of Squares using clustering results
'''
def plot_wcss(X):   
    wcss = []

    # trying kmeans for k=1 to k=10
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(X=X)
        wcss.append(kmeans.inertia_)

    sns.set(style='darkgrid')
    sns.lineplot(x=range(1,11), y=wcss)    
    plt.title('Finding Optimal No. of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()


'''
Performs K-Means clustering
'''
def kmeans(X, n_clusters):
    # apply kmeans to the dataset
    algo = KMeans(n_clusters=n_clusters, random_state=10)
    
    # gives us cluster-assignment for each row
    clusters = algo.fit_predict(X)

    return clusters


    

'''
Main Program
'''

# read dataset into a Pandas DataFrame
df = pd.read_csv('iris.csv')

# removing Id column as it is just a running number 
# and does not add any value to our analysis.
df.drop(['Id'], axis=1, inplace=True)

'''
Examining our data.
'''

# display Pandas DataFrame
print(df)

# display the statistical summary of the dataset
print(df.describe())

# display statistical summary by 'Species'
df[df['Species']=='Iris-setosa'].describe()
df[df['Species']=='Iris-versicolor'].describe()
df[df['Species']=='Iris-virginica'].describe()

# plot distribution within our dataset
plot_dist_by_attr(df)

# plot distribution by Species
kde_plot(df)


'''
Performs K-Means clustering.
'''
X = df.drop(['Species'], axis=1)    # 'label' is not needed for clustering
X = X.values                        # convert to NumPy array
clusters = kmeans(X, n_clusters=3)
print('Clustering Results: {0}'.format(clusters))

# note that we have 150 rows of data, hence 150 rows of cluster-assignment
# each row of cluster-assignment maps to a row of data
# for example, if clusters[0] == 0, then X[0] has been assigned to cluster 0
print('Len: {0}'.format(len(clusters)))

# visualize the clustering results
# plot_cluster_results(X=X, clusters=clusters)


'''
Plot WCSS to locate elbow, which tells us the optimal number of clusters
'''
plot_wcss(X=X)