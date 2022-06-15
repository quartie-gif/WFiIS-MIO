import random

import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


from EcgAutoencoder import EcgAutoencoder
from load_data import load_ecg_data


def generate_graph(x_train, x_test, layers, encoder_output_index, epoch_count, batch_size, index):
    # # Initialize model
    input_size = x_train.shape[1]
    layers = layers
    layer_sizes = [tup[0] for tup in layers]
    activations = [tup[1] for tup in layers]
    loss = 'binary_crossentropy'
    model = EcgAutoencoder(
        input_size=input_size,
        layer_sizes=layer_sizes,
        activations=activations,
        encoder_output_index=encoder_output_index,
        loss=loss
    )
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    steps_per_epoch = len(x_train) // batch_size
    # # Fit model
    history = model.autoencoder.fit(x_train, x_train,
                                    epochs=epoch_count,
                                    steps_per_epoch=steps_per_epoch,
                                    batch_size=batch_size,
                                    callbacks=[callback],
                                    shuffle=True,
                                    validation_data=(x_test, x_test))
    model.build(input_shape=(None, input_size))

    encoded_ecg = model.encoder.predict(x_test)
    decoded_ecg = model.decoder.predict(encoded_ecg)

    # # Visualize results
    n = 5
    # Choose random inputs to visualize
    indices = [random.randint(0, len(x_test)-1) for _ in range(n)]
    plt.figure(figsize=(n * 3, 4))
    for vis_i, i in enumerate(indices):
        # Display original
        ax = plt.subplot(3, n, vis_i + 1)
        plt.plot(range(len(x_test[i])), x_test[i], color="#4D97B2")
        plt.title("Original data")

        # Display encoded
        ax = plt.subplot(3, n, vis_i + 1 + n)
        plt.plot(range(len(encoded_ecg[i])), encoded_ecg[i], color="#B24D97")
        plt.title("Encoded data")

        # Display reconstructed
        ax = plt.subplot(3, n, vis_i + 1 + 2 * n)
        plt.plot(range(len(decoded_ecg[i])), decoded_ecg[i], color="#97B24D")
        plt.title("Decoded data")

    plt.tight_layout()
    plt.savefig(f'./images/results_{index}.png', dpi=300, bbox_inches='tight')

    # summarize history for accuracy
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over time')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Loss on train data', 'Loss on test data'], loc='upper left')
    plt.grid()
    print("Compression={}".format(len(x_test[i]) / len(encoded_ecg[i])))
    plt.savefig(f'loss_{index}.png'
                , dpi=300
                , bbox_inches='tight')


def main():
    (x_train, x_test, _, _) = load_ecg_data()
    layerss = [[(16, 'relu'), (32, 'relu'), (16, 'relu')],
              [(32, 'relu'), (16, 'relu'), (32, 'relu')],
              [(64, 'relu'), (64, 'relu'), (64, 'relu')],
              [(8, 'relu'), (8, 'relu'), (16, 'relu')],
              [(32, 'relu'), (16, 'relu'), (32, 'relu')]]
    encoder_output_indices = [1, 1, 1, 1, 1]
    epoch_counts = [150, 150, 150, 150, 150]  # [30, 12, 40, 36, 100]
    batch_sizes = [100, 256, 256, 256, 256]
    for i, (layers, encoder_output_index, epoch_count, batch_size) in enumerate(zip(layerss, encoder_output_indices, epoch_counts, batch_sizes)):
        print(f"Model {i}:")
        generate_graph(x_train, x_test, layers, encoder_output_index, epoch_count, batch_size, i)


if __name__ == '__main__':
    main()
