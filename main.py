import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import cv2
import csv
warnings.filterwarnings('ignore')  # to suppress the warnings
import Autoencoder

for dirname, _, filenames in os.walk('/dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


MIT_OUTCOME = {0.: 'Normal Beat',
                   1.: 'Supraventricular premature beat',
                   2.: 'Premature ventricular contraction',
                   3.: 'Fusion of ventricular and normal beat',
                   4.: 'Unclassifiable beat'}


def autoencoding_model(X_train, X_test):
    pass
    # create a model


def temp():
    # TODO
    pass

def preprocess_data_plot(X_train, X_test):

    X_test.rename(columns={187: "Class"}, inplace=True)
    X_train.rename(columns={187: "Class"}, inplace=True)

    return X_test, X_train

def preprocess_data(X_train, X_test):

    X_test.rename(columns={187: "Class"}, inplace=True)
    X_train.rename(columns={187: "Class"}, inplace=True)
    X_train.drop(columns=['Class'], inplace=True)
    X_test.drop(columns=['Class'], inplace=True)
    print(X_train.shape)
    print(X_test.shape)
    return X_test, X_train


def plot_image(mit_train):

    plt.figure(figsize=(25, 10))
    np_count = np.linspace(0, 186, 187)
    np_time = np.tile(np_count, (10, 1))
    rnd = np.random.randint(0, mit_train.shape[0], size=(10,))

    for i in range(np_time.shape[0]):
        ax = plt.subplot(2, 5, i+1)
        ax.plot(mit_train[rnd[i]])
        # ax.set_title(MIT_OUTCOME[])

    # plt.show()
    plt.savefig("out.png")


def main():
    mit_test = pd.read_csv('./dataset/mitbih_test.csv', header=None)
    mit_train = pd.read_csv('./dataset/mitbih_train.csv', header=None)
    mit_test, mit_train = preprocess_data(mit_test, mit_train)
    # plot_image(mit_train)
    x_axis, y_axis = Autoencoder.test(mit_train, mit_test)
    plot_image(y_axis)



if __name__ == "__main__":
    main()
