import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


def test(
        x_train,
        x_test):

    encoding_dim = 32

    input_img = keras.Input(shape=(187,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    decoded = layers.Dense(187, activation='sigmoid')(encoded)

    autoencoder = keras.Model(input_img, decoded)
    print(autoencoder.summary())

    encoder = keras.Model(input_img, encoded)

    encoded_input = keras.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # x_train = x_train.astype('float32')/255
    # x_test = x_test.astype('float32')/255

    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    print(x_train.shape)
    print(x_test.shape)

    # autoencoder.fit(x_train, x_train,
    #                 epochs=12,
    #                 batch_size=256,
    #                 shuffle=True,
    #                 validation_data=(x_test, x_test))

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10  # How many digits we will display
    plt.figure(figsize=(n*3, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        x_axis = range(len(np.array(x_test[i])))
        plt.plot(x_axis, np.array(x_test[i]))
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # Display encoded
        # ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(encoded_imgs[i].reshape(4, 8))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        # Display reconstruction
        # ax = plt.subplot(2, n, i + 1 + 2*n)
        # plt.imshow(decoded_imgs[i].reshape(28, 28))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    plt.savefig("asdasd.png")
    y_axis=np.array(x_test)
    return x_axis, y_axis
