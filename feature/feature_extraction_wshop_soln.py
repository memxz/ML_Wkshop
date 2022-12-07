import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


'''
Plot feature distributions 
'''
def plot_distr(df, title):
  sns.set_style(style="darkgrid")

  # create a grid of plots of nrows and ncols
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

  sns.kdeplot(data=df, x=df['Alcalinity of ash'], 
      hue='Cultivar', # 'hue' allows us to differentiate by type/label
      ax=ax[0, 0], palette='tab10')  

  sns.kdeplot(data=df, x=df['Flavanoids'], 
      hue='Cultivar', # 'hue' allows us to differentiate by type/label
      ax=ax[0, 1], palette='tab10')  

  sns.kdeplot(data=df, x=df['Hue'], 
      hue='Cultivar', # 'hue' allows us to differentiate by type/label
      ax=ax[1, 0], palette='tab10')  

  sns.kdeplot(data=df, x=df['Proline'], 
      hue='Cultivar', # 'hue' allows us to differentiate by type/label
      ax=ax[1, 1], palette='tab10')  

  fig.suptitle(title)
  plt.show()



'''
Perform feature scaling / standardization.
Expects provided data to be a NumPy array.
Returns scaled dataset.
'''
def scale_features(data_np):
  return StandardScaler().fit_transform(X=data_np)


'''
Performs PCA on data. The features for the provided dataset 
must already be scaled.
Expects input data to be a NumPy array.
Returns the pca object and the newly-generated features 
'''
def apply_pca(n_components, data_np):
  pca = PCA(n_components=n_components)
  return pca, pca.fit_transform(X=data_np)


'''
Construct a Pandas Dataframe from 'data_np' while using 'ref_data_df'
as a reference (e.g. feature_names and label)
Assumes the label is not found in 'data_np'.
'''
def make_dataframe_from(ref_data_df, data_np, label):   
  data_df = pd.DataFrame(
    data=data_np, # columns of data to fit into reference dataframe
    columns=ref_data_df.columns.drop(label) # want all columns but the label
  )

  # copied the label over to new dataframe
  data_df[label] = ref_data_df[label]
  return data_df


'''
Determines the minimum number of principal components to capture
X% ('percent' parameter) of the original information in our dataset.
'''
def min_components(data_np, percent):
  min = 1

  # shape[1] gives us the number of columns in our dataset
  # starts from 1 to leave out first column (Cultivar, which is our label)
  # +1 to include the last column
  for min in range(1, data_np.shape[1] + 1):
    pca = PCA(n_components=min)
    pca.fit(data_np)  # did not transform (i.e. generate new features)
    if pca.explained_variance_ratio_.sum() >= percent:
      return min
    


'''
Main Program
'''

# reading in our dataset
data_df = pd.read_csv("wine.csv")
print(data_df)

# plot distributions before scaling features  
plot_distr(df=data_df, title='Before Feature-Scaling')

# transform dataframe into numpy array.
# use all columns except 'Cultivar' (our label) since 
# there is no need to scale our label.
data_np = data_df.loc[:, "Alcohol":].values

# scale the features in the numpy array
scaled_data_np = scale_features(data_np=data_np)

# convert our scaled numpy data back into a dataframe for plotting distributions
scaled_data_df = make_dataframe_from(
  ref_data_df=data_df, 
  data_np=scaled_data_np, 
  label='Cultivar'
)

# plot distributions after scaling features  
plot_distr(df=scaled_data_df, title='After Feature-Scaling')

# no need for the new features here, hence the second return
# value is a '_'
pca, _ = apply_pca(n_components=4, data_np=scaled_data_np)

# explained variance ratios capturd by each of the 4 principal components
print("Explained Variance Ratios =", pca.explained_variance_ratio_)

# total explained variance ratios by all 4 principal components
print("Total Explained Variance =", pca.explained_variance_ratio_.sum())

# determine the no. of principal components required to capture
# at least 85% of the information of the original dataset
min_pc = min_components(data_np=scaled_data_np, percent=0.85)
print("Min principal components =", min_pc)
