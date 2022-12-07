import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32,
        kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))    
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

paths = []

for i in range(10):
    paths.append('./mnist_train/{}/'.format(i))

for i in range(len(paths)):
    for file in os.listdir(paths[i]):
        if file[0] == '.':  # skip hidden files
            continue

        # reading image file into memory
        img = Image.open("{}/{}".format(paths[i], file))

        try:
            x_train = np.concatenate((x_train, img))
        except:
            x_train = img   

    # image is 28x28 and gray-scale, hence there
    # is only 1 channel (28, 28, 1).
    # -1 to let numpy computes the number of rows 
    np.reshape(x_train, (-1, 28,28,1))     

    try:
        x = np.concatenate((x, x_train))
    except:
        x = x_train          
    # construct the onehot-encodings for a digit's data
 
    # 10 classes (digit 0 to 9)
    y_onehot = [0] * 10
    # create onehot-encodings for digit (i - 1)
    y_onehot[i] = 1
    y_onehots = [y_onehot] * x.shape[0]
    # convert python list to numpy array
    # as keras requires numpy array
    np.array(y_onehots)
    try:
        y = np.concatenate((y, y_onehots))
    except:
        y = y_onehots    

print(x,x_train,y_onehot)       