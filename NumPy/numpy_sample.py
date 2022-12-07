import numpy as np

# NOTE: The functions below assume inputs as 2d NumPy arrays.


# return min inside the matrix
def min_value(mat):
    return mat.min()
    

# return row that yields min-sum by row 
def max_value(mat):
    return mat.max()


# return max inside the matrix
def min_row(mat):
    # compute sum of each row
    sum_by_rows = mat.sum(axis=1)       
    # get index of row that yields min-sum
    row_idx = sum_by_rows.argmin()      
    # returns row that yields the min-sum
    return mat[row_idx]                 


# return row that yields max-sum by row
def max_row(mat):
    # compute sum of each row
    sum_by_rows = mat.sum(axis=1)       
    # get index of row that yields max-sum
    row_idx = sum_by_rows.argmax()      
    # returns row that yields the max-sum
    return mat[row_idx]                 
    

# return a row using the min value from each row
# in the matrix
def min_from_each_row(mat):
    # collapsing columns
    return mat.min(axis=1)


# return a row using the max value from each row
# in the matrix
def max_from_each_row(mat):
    # collapsing columns
    return mat.max(axis=1)


# return a row using the min value from each column
def min_from_each_column(mat):
    # collapsing rows
    return mat.min(axis=0)


# return a row using the max value from each column
def max_from_each_column(mat):
    # collapsing rows
    return mat.max(axis=0)


# return a 1d array in descending order 
def flatten_max_first(mat):
    # reshape into a 1d array
    # -1 means as many columns as needed
    flat = mat.reshape(1, -1)

    # converting numpy array into a python list
    # flat[0] to turn into 1d array
    arr = flat[0].tolist()

    # in-place sorting taking place
    arr.sort(reverse=True)  
    
    return arr


# return the four-corner values of a 2d numpy array
def four_corner_values(mat):
    return [
        mat[0, 0],  # top-left
        mat[0, -1], # top-right
        mat[-1, 0], # bottom-left
        mat[-1, -1] # bottom-right
    ]


# set all values below a threshold to value 'to_x'
def filter_mat(mat, threshold, to_x):
    # generate a matrix of booleans that match the 
    # criteria of (mat < threshold)
    tf_mat = mat < threshold
    
    # use the matrix of booleans to match
    # the rows, in mat, to change
    mat[tf_mat] = to_x
    return mat



'''
Examples of Python with NumPy.
'''
print()

# randomly generating a matrix of values between 0 and 1.0
mat = np.random.rand(4, 3)
print('mat =', mat, '\n')

# Example 1
print('min value:', min_value(mat), '\n')

# Example 2
print('max value:', max_value(mat), '\n')

# Example 3
print('min row:', min_row(mat), '\n')

# Example 4
print('max row:', max_row(mat), '\n')

# Example 5
print('min_from_each_row:', min_from_each_row(mat), '\n')

# Example 6
print('max_from_each_row:', max_from_each_row(mat), '\n')

# Example 7
print('min_from_each_column:', min_from_each_column(mat), '\n')

# Example 8
print('max_from_each_column:', max_from_each_column(mat), '\n')

# Example 9
print('flatten_max_first:', flatten_max_first(mat), '\n')

# Example 10
print('four_corner_values:', four_corner_values(mat), '\n')

# Example 11
print('filter_mat:', filter_mat(mat, 0.15, 0.0), '\n')
