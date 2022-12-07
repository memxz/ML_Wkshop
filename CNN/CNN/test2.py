import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

img = Image.open('./mnist_train/0/img_1.jpg')
img_data = np.array(img)
# print(img_data)
print(img_data.shape)
plt.imshow(img_data, cmap='gray')
# plt.show()