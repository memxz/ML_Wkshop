import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

'''
Sample code to show how to use PCA in scikit-learn (sklearn).
'''

df = pd.read_csv('iris.csv')
data = df.values

# get all rows; exclude first and last column only those columns 
# are considered 'features' to us
x = data[:,1:-1]
print(x)

# use StandardScaler to zero-mean and unit-variant on each extracted 
# column (note that each column represents multiple observations of 
# a single feature in our dataset)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# pass standardized data to PCA out of the 4 existing features,
# we want to create 2 NEW features
pca = PCA(n_components=2)

# returns a NumPy array of size 'n_components' 'pc' contains 
# the new features generated applying PCA on the dataset
pc = pca.fit_transform(x_scaled)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())