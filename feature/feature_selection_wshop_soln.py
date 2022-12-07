import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


'''
Create a pair-wise correlation plot.
'''
def make_pairplot(df, hue_target):
  sns.set(style='darkgrid')
  sns.pairplot(df, hue=hue_target, height=1, aspect=1.6, palette='tab10')
  plt.show()



'''
Generate a Heat Map from a correlational matrix and
save it to a file.
'''
def make_heatmap(corr_mat):
  plt.figure(figsize=(12,10))
  sns.heatmap(data=corr_mat,     # correlation matrix
    annot=True,             # display pearson correlation values
    annot_kws={'size':8},   # font size for values
    cmap='GnBu')
  plt.show()



'''
Performs feature-selection. It selects the best features by looking
for features that correlates strongly w.r.t. the label. It also
removes redunduncy by eliminating peers that are strongly-correlated.
'''
def best_features(corr_mat, label, label_limit, peers_limit):
  
  # only consider features that are highly-correlated with our label
  candidates_df = corr_mat.loc[
    (corr_mat[label] < -label_limit) | (corr_mat[label] > label_limit),   # index
    (corr_mat[label] < -label_limit) | (corr_mat[label] > label_limit)    # column
  ] 

  # move our 'label' column to the end for easier processing later
  candidates_df = candidates_df.drop(columns=[label]) # remove 'label' from the columns
  candidates_df[label] = corr_mat[label] # move 'label' column to the last column
  
  # use this to store the best features so far
  accept = [] 

  # iteratively compares features against peers and the label
  while len(candidates_df.columns) > 1:  # stop when left with only our label
    # inspect each feature in turn
    feature = candidates_df.columns[0]

    # get all peers of these feature, except the label
    peers = candidates_df.loc[feature].drop(label)
    print("peers of '", feature, "' =\n", peers, '\n', sep='')

    # look for other features that are highly-correlated with 
    # the current 'feature'. only consider positively correlated 
    # to remove redundancy.
    high_corr = peers[peers > peers_limit]   
    print('high_corr =\n', high_corr, '\n', sep='')

    # extract the pearson correlation values of each of these 
    # highly-correlated features w.r.t. our label
    alike = candidates_df.loc[high_corr.index, label]
    print('alike =\n', alike, '\n', sep='')

    # idxmax() to get the feature that is most correlated with 
    # our label. abs() to absolute the values because 
    # the features could be either positively or negatively 
    # correlated to our label
    top = alike.abs().idxmax()  # row-label (feature-name) of max-value
    accept.append(top)

    # place index-names into an array, so that we can use them
    # as parameters to the drop() function
    alike = list(alike.index) 

    # done with feature, remove feature from 'candidates_df',
    # this allows our candidates to be smaller each time
    candidates_df = candidates_df.drop(columns=alike, index=alike)

  # returns best features in the    
  return accept

  
'''
Main Program
'''

# read in our dataset
df = pd.read_csv('wine.csv')
print(df)

# generate plots
make_pairplot(df=df, hue_target='Cultivar')

# generate correlation matrix and heatmap
corr_mat = df.corr()
make_heatmap(corr_mat=corr_mat)

print('Starting Feature Selection ...')
# feature selection - find the best features that correlates 
# to our label; use these features to train our model for
# future predictions
best = best_features(corr_mat=corr_mat, label='Cultivar', label_limit=0.5, peers_limit=0.6)

print('best features =', best)
