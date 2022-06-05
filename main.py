import random

import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

from EcgAutoencoder import EcgAutoencoder
from load_data import load_ecg_data


def main():
    # # Load data
    x_train, x_test, _, _ = load_ecg_data()
    print(x_train.shape)
    print(x_test.shape)

    # # Initialize model
    input_size = x_train.shape[1]
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

    # # Fit model
    model.autoencoder.fit(x_train, x_train,
                          epochs=12,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

    encoded_ecg = model.encoder.predict(x_test)
    decoded_ecg = model.decoder.predict(encoded_ecg)

    # # Visualize results]
    n = 5
    # Choose random inputs to visualize
    indices = [random.randint(0, len(x_test)) for _ in range(n)]
    plt.figure(figsize=(n * 3, 4))
    for vis_i, i in enumerate(indices):
        # Display original
        ax = plt.subplot(3, n, vis_i + 1)
        plt.plot(range(len(x_test[i])), x_test[i])

        # Display encoded
        ax = plt.subplot(3, n, vis_i + 1 + n)
        plt.plot(range(len(encoded_ecg[i])), encoded_ecg[i])

        # Display reconstructed
        ax = plt.subplot(3, n, vis_i + 1 + 2 * n)
        plt.plot(range(len(decoded_ecg[i])), decoded_ecg[i])

    plt.tight_layout()
    plt.savefig('test.png')


if __name__ == '__main__':
    main()
