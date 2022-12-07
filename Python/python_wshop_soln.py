# O(n^2) time to look for two indices whose values add to target n
def quad_find(s, n):
    slen = len(s)

    for i in range(slen):
        for j in range(slen):
            if s[i] + s[j] == n:
                return (i, j)

    return None


# O(n) time to look for two indices whose values add to target n
def linear_find(s, n):    
    dict = {}

    for i in range(len(s)):
        if s[i] in dict:
            return (i, dict[s[i]])    # s[i] contains the other index
        else:
            complement = n - s[i]
            dict[complement] = i
    
    return None


s = [10, 3, -5, 7, -9, 0, -2, 5]
print(quad_find(s, 7))
print(linear_find(s, 7))