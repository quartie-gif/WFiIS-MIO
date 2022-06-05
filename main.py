import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

from EcgAutoencoder import EcgAutoencoder


def main():
    # # Initialize model
    input_size = 784
    layers = [(32, 'relu'),
              (16, 'relu'),
              (32, 'relu')]
    layer_sizes = [tup[0] for tup in layers]
    activations = [tup[1] for tup in layers]
    encoder_output_index = 1
    loss = 'binary_crossentropy'
    model = EcgAutoencoder(
        input_size=input_size,
        layer_sizes=layer_sizes,
        activations=activations,
        encoder_output_index=1,
        loss=loss
    )

    # # Load data
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    # # Fit model
    model.autoencoder.fit(x_train, x_train,
                          epochs=12,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

    encoded_imgs = model.encoder.predict(x_test)
    decoded_imgs = model.decoder.predict(encoded_imgs)

    # # Visualize results
    n = 10
    plt.figure(figsize=(n * 3, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display encoded
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs[i].reshape(4, 4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('test.png')


if __name__ == '__main__':
    main()
