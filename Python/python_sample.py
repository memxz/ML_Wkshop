# find all locations of a character in string
def find_char_pos(str, ch):
    pos = []    # array

    for i in range(len(str)):
        if str[i] == ch:
            pos.append(i)
    
    return pos


# count no. of occurences of each character in string
def count_chars(str):
    chars = {}  # dictionary

    for ch in str:
        if ch in chars:
            chars[ch] += 1
        else:
            chars[ch] = 1
    
    return chars


# compute string length recursively
def recursive_strlen(s):
    if len(s) == 0:
        return 0

    return 1 + recursive_strlen(s[1:])



# Example 1
s = "Let your life be shaped by decisions you made, not by the ones you didn't"
ch = 'e'
pos = find_char_pos(s, ch)

if len(pos) == 0:
    print("Character '{0}' is not found in the given string".format(ch))
else:
    print("Character '{0}' is found at index {1}".format(ch, pos))
print()


# Example 2
s = "Every time we open our mouths, men look into our minds"
print('Occurences: {0}'.format(count_chars(s)))
print()


# Example 3
s = 'hello'
print("The string '{0}' has {1} characters.".format(s, recursive_strlen(s)))
print()

