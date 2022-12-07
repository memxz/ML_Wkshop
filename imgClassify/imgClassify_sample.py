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
def show_sample_digit():
    img = Image.open('./train/apple_1.jpg')
    img_blur=img.filter(ImageFilter.BLUR)
    plt.imshow(img, cmap='gray')
    plt.show()

'''
Performs onehot-encodings for every digit (0 to 9).
'''
def encode_onehot(pos, n_rows):
    # 10 classes (digit 0 to 9)
    y_onehot = [0] * 4
    # create onehot-encodings for digit (i - 1)
    y_onehot[pos] = 1
    y_onehots = [y_onehot] * n_rows
    # convert python list to numpy array
    # as keras requires numpy array
    return np.array(y_onehots)

'''
Read image data
'''
def read_img_data(path):
    for file in os.listdir(path):
        if file[0] == '.':  # skip hidden files
            continue

        # reading image file into memory
        img = Image.open("{}/{}".format(path, file))
        img = img.convert('RGB')
        img = img.resize((50,50))
        try:
            x_train = np.concatenate((x_train, img))
        except:
            x_train = img   

    # image is 28x28 and gray-scale, hence there
    # is only 1 channel (28, 28, 1).
    # -1 to let numpy computes the number of rows 
    return np.reshape(x_train, (-1, 50,50,3))     
   
'''
Prepare data.
'''
def prep_data(paths):
    for i in range(len(paths)):
        data = read_img_data(paths[i])

        try:
            x = np.concatenate((x, data))
        except:
            x = data          

        # construct the onehot-encodings for a digit's data
        y_onehots = encode_onehot(i, data.shape[0])
        try:
            y = np.concatenate((y, y_onehots))
        except:
            y = y_onehots           

    return x, y

'''
Prepare train data.
'''
def prep_train_data():
    paths = []

    for i in range(4):
        paths.append('./train/{}/'.format(i))

    return prep_data(paths)

'''
Prepare test data.
'''
def prep_test_data():
    paths = []

    for i in range(4):
        paths.append('./test/{}/'.format(i))

    return prep_data(paths)

'''
Create our model
'''
def create_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32,
        kernel_size=(3, 3), activation='relu', input_shape=(50,50,3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))    
    model.add(tf.keras.layers.Dense(units=4, activation='softmax'))    
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                    metrics=['accuracy'])

    return model

'''
Train our model.
'''
def train_model(model, x_train, y_train):
    model.fit(x=x_train, y=y_train, epochs=20)    

'''
Test our model.
'''
def test_model(model, x_test, y_test):
    return model.evaluate(x=x_test, y=y_test)

'''
Save our model.
'''
def save_model(model, path):
    model.save(path)

'''
Load our model.
'''
def load_model(path):
    return tf.keras.models.load_model(path)

'''
Main program.
'''
def main():
    # create our CNN model
    model = create_model()

    # fetch training data and onehot-encoded labels
    x_train, y_train = prep_train_data()
    
    # normalize x_train to be between [0, 1]
    train_model(model, x_train/255, y_train)

    # showing how we can save our trained model
    save_model(model, './image_saved_model')
    
    # showing how we can load our trained model
    model = load_model('./image_saved_model')

    # normalize y_train to be between [0, 1]
    x_test, y_test = prep_test_data()

    # test how well our model performs against data
    # that it has not seen before
    test_model(model, x_test/255, y_test)


# running via "python mnist_sample.py"
if __name__ == '__main__':
  main()


