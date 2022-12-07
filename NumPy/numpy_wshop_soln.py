import numpy as np

# count occurences of value 'val' in the numpy array
def count_occurences(mat, val):
    # generate boolean (true/false) matrix based
    # on given condition.
    tf_mat = (mat == val)

    # count the no. of 'True' in the matrix
    return np.count_nonzero(tf_mat)


# checks if any rows, columns and diagonals sums
# to the value 'val'. 
def sums_to_value(mat, val):
    # must be 2d array
    if mat.ndim != 2:
        return None

    # must be NxN array
    if mat.shape[0] != mat.shape[1]:
        return None

    for i in range(2):
        # to collapse by column (axis=0)
        # to collapse by row (axis=1)
        arr = mat.sum(axis=i)
        tf_mat = (arr == val)

        # checks if any value in the boolean array
        # has the value True
        if tf_mat.any():
            return True

    # checks diagonal-sums
    l2r_sum = 0
    r2l_sum = 0

    for i in range(0, mat.shape[0]):
        # moving from top-left to bottom-right
        l2r_sum += mat[i, i]    
        # moving from bottom-right to top-left
        r2l_sum += mat[mat.shape[0]-1-i, i]
    return (l2r_sum == val) or (r2l_sum == val)


# given 2d numpy array
mat = np.array([
    [3, 5, -6, 8],
    [3, 1, 6, -5],
    [8, -5, 3, 8],
    [7, 8, 1, -2]
])

print('occurences:', count_occurences(mat, 8))
print('sums to value:', sums_to_value(mat, 5))

