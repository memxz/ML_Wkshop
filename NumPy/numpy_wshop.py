import numpy as np

'''
Implement the two Python functions.
'''

# count occurences of value 'val' in the numpy array
def count_occurences(mat, val):
    tf_mat=(mat==val)
    return np.count_nonzero(tf_mat)


# checks if any rows, columns and diagonals sums
# to the value 'val'. 
def sums_to_value(mat, val):
    if(mat.ndim!=2):
        return None

    if mat.shape[0]!=mat.shape[1]:
        return None

    for i in range(2):
        arr=mat.sum(axis=i)
        tf_mat=(arr==val)
        
        if tf_mat.any():
            return True

    l2r_sum=0
    r2l_sum=0
    for i in range(0, mat.shape[0]):
        # moving from top-left to bottom-right
        l2r_sum += mat[i, i]    
        # moving from bottom-left to top-right
        r2l_sum += mat[mat.shape[0]-1-i, i]
    return (l2r_sum == val) or (r2l_sum == val)
   # print(l2r_sum,r2l_sum)


# given 2d numpy array
mat = np.array([
    [3, 5, -6, 8],
    [3, 1, 6, -5],
    [8, -5, 3, 8],
    [7, 8, 1, -2]
])

print('occurences:', count_occurences(mat, 8))
print('sums to value:', sums_to_value(mat, 16))
