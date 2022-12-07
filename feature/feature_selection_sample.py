import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


'''
Create a pair-wise correlation plot.
'''
def plot_pairwise(df):
  sns.set(style='darkgrid')
  sns.pairplot(df, hue='origin', height=1.5, aspect=1.0, palette='tab10')
  plt.show()
  #plt.savefig('plot.png')  # save to file
  #plt.clf() # clear plot; next plot starts afresh



'''
Main Program
'''

# read in our data
df = pd.read_csv('auto-mpg.csv')
print(df, '\n')

# compute pearson correlation matrix
corr_mat = df.corr()
print(corr_mat, '\n')

# plot pair-wise correlation
plot_pairwise(df)


'''
Performing Feature Selection
'''

# select our label/target
label = 'mpg'

# only consider features that are highly-correlated with our label
# using threshold as -0.5 to +0.5
candidates_df = corr_mat.loc[
  (corr_mat[label] < -0.5) | (corr_mat[label] > 0.5),   # index
  (corr_mat[label] < -0.5) | (corr_mat[label] > 0.5)    # column
] 

# move our 'label' column to the end for easier processing later
candidates_df = candidates_df.drop(columns=[label]) # remove 'mpg' from the columns
candidates_df[label] = corr_mat[label] # move 'mpg' column to the last column

# after moving 'mpg' column to the end
print(candidates_df, '\n')

# use this to store the best features so far
accept = [] 

# iteratively compares features against peers and the label
while len(candidates_df.columns) > 1:  # stop when left with only our label
  # inspect each feature in turn
  feature = candidates_df.columns[0]

  # get all peers of these feature, except the label
  peers = candidates_df[feature].drop(label)
  print("peers of '", feature, "' =\n", peers, '\n', sep='')

  # look for other features that are highly-correlated with 
  # the current 'feature'. only consider positively correlated 
  # to remove redundancy.
  high_corr = peers[peers > 0.6]   # using threshold of 0.6 to filter
  print('high_corr =\n', high_corr, '\n', sep='')

  # extract the pearson correlation values of each of these 
  # highly-correlated features w.r.t. our label
  alike = candidates_df.loc[high_corr.index, label]
  print('correlation w.r.t label =\n', alike, '\n', sep='')

  # idxmax() to get the feature that is most correlated with 
  # our label. abs() to get the maximum values, regardless negative
  # or positive values.
  top = alike.abs().idxmax()  # row-label (feature-name) of max-value
  accept.append(top)

  # place index-names into an array, so that we can use them
  # as parameters to the drop() function
  alike = list(alike.index) 

  # done with feature, remove feature from 'candidates_df',
  # this allows our candidates to be smaller each time
  candidates_df = candidates_df.drop(columns=alike, index=alike)

  # after dropping features that we have processed
  print(candidates_df, '\n')

# features selected using Feature Selection
print('Selected Features = ', accept)