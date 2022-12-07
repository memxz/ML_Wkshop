import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageFilter


# img = Image.open('./train/0/img_1.jpg')
# img_data = np.array(img)
# print(img_data)
# print(img_data.shape)
# plt.imshow(img_data, cmap='gray')
# plt.show()


'''
Show an sample digit.
'''
img = Image.open('./train/0/apple_10.jpg')
img_blur=img.filter(ImageFilter.BLUR)
plt.imshow(img, cmap='gray')
plt.show()