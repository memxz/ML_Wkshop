from venv import create
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



'''
Splits our data into train and test sets.
'''
def train_test_data(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, stratify=labels
    )

    return x_train, x_test, y_train, y_test


'''
Perform zero-mean and unit-variance standardization.
'''
def standardize_data(x_train, x_test):
    scaler = StandardScaler()

    # fit our scaler on the training data
    scaler.fit(x_train)

    # transform our train and test data on the fitted weights
    x_train_std = scaler.transform(x_train)
    x_test_std = scaler.transform(x_test)

    return x_train_std, x_test_std


'''
Perform dimension reduction using PCA.
'''
def reduce_dim(x_train, x_test):
    pca = PCA(n_components=5)

    # again, fit on training data
    pca.fit(x_train)

    # transform on both train and test data
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    return x_train_pca, x_test_pca


'''
Perform one-hot encoding on a list of labels.
'''
def to_onehot(arr):
    onehot = []

    for label in arr:
        if label == 1:
            _1hot = [1, 0, 0]
        elif label == 2:
            _1hot = [0, 1, 0]
        else:   # label == 3
            _1hot = [0, 0, 1]
        
        onehot.append(_1hot)

    # converting from python list to numpy array
    return np.array(onehot)


'''
Create our Neural Network model.
'''
def create_model(n_features):
    model = tf.keras.Sequential()

    # add a layer with 100 neurons
    model.add(tf.keras.layers.Dense(units=100, 
        input_shape=(n_features,), activation='relu'))

    # 'softmax' becauses we are doing classification.
    # a sample can only fall into one of the 3 classes. 
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    # use 'categorical_crossentropy' for our loss calcuation
    model.compile(optimizer='adam', loss='categorical_crossentropy', 
        metrics=['accuracy'])
    
    return model



'''
Train our model.

Returns the 'history' of our training, which has the loss 
and accuracy details for each epoch.
'''
def train_model(model, x_train, y_train_1hot):
    return model.fit(x_train, y_train_1hot, epochs=300)


'''
Create loss and accuracy plots.
'''
def plot(hist):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(hist.history['loss'])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')

    ax[1].plot(hist.history['accuracy'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')

    plt.show()    


'''
Automatic evaluation of our model against test set
'''
def auto_eval(model, x_test, y_test_1hot):
    loss, accuracy = model.evaluate(x=x_test, y=y_test_1hot)

    print('loss = ', loss)
    print('accuracy = ', accuracy)    


'''
Do our own evaluation; printing out predictions given by our model.
'''
def manual_eval(model, x_test, y_test_1hot):
    # get predicted values from model
    predictions = model.predict(x=x_test)

    # eyeball predicted values against actual ones
    for i in np.arange(len(predictions)):
        print('Actual: ', y_test_1hot[i], 'Predicted: ', predictions[i])        

    # compute accuracy
    n_preds = len(predictions)       
    correct = 0
    wrong = 0

    for i in np.arange(n_preds):
        pred_max = np.argmax(predictions[i])
        actual_max = np.argmax(y_test_1hot[i])
        if pred_max == actual_max:
            correct += 1
        else:
            wrong += 1
    
    print('correct: {0}, wrong: {1}'.format(correct, wrong))
    print('accuracy =', correct/n_preds)

        

'''
Main Program.
'''
def main():
    df = pd.read_csv('wine.csv')
    print(df)

    # get all rows and all columns from 'Alcohol' onwards;
    # .values to convert to numpy array
    features = df.loc[:, 'Alcohol':].values

    # get our labels
    labels = df[['Cultivar']].values

    # split our data into train and test sets
    x_train, x_test, y_train, y_test = train_test_data(features, labels)

    # standardize our data
    x_train_std, x_test_std = standardize_data(x_train, x_test)

    # reduce dimension
    x_train_pca, x_test_pca = reduce_dim(x_train_std, x_test_std)

    # perform one-hot encoding
    y_train_1hot = to_onehot(y_train)
    y_test_1hot = to_onehot(y_test)

    # create and train our model       
    model = create_model(
        x_train_pca.shape[1]    # no. of features used for training 
    ) 
    hist = train_model(model, x_train_pca, y_train_1hot)

    # loss and accuracy plots
    plot(hist)

    # tensorflow will do auto-evaluation of model against test set
    auto_eval(model, x_test_pca, y_test_1hot)

    # perform manual evaluation of our model using test set
    manual_eval(model, x_test_pca, y_test_1hot)



if __name__ == "__main__":
  main()
