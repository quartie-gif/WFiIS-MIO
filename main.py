import random

import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz


from EcgAutoencoder import EcgAutoencoder
from load_data import load_ecg_data


def main():
    # # Load data
    x_train, x_test, _, _ = load_ecg_data()
    print(x_train.shape)
    print(x_test.shape)

    # # Initialize model
    input_size = x_train.shape[1]
    layers = [(256, 'relu'),
              (128, 'relu'),
              (64, 'relu'),
              (128, 'relu'),
              (256, 'relu')]
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

    # Early stopping callback

    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    batch_size = 16
    steps_per_epoch = len(x_train)//batch_size
    # # Fit model
    history = model.autoencoder.fit(x_train, x_train,
                                    epochs=50,
                                    steps_per_epoch=steps_per_epoch,
                                    batch_size=batch_size,
                                    callbacks=[callback],
                                    shuffle=True,
                                    validation_data=(x_test, x_test))
    # ann_viz(model, title="Autoencoder")
    model.build(input_shape=(None, input_size))
    # print(model.summary())

    encoded_ecg = model.encoder.predict(x_test)
    decoded_ecg = model.decoder.predict(encoded_ecg)

    # # Visualize results
    n = 5
    # Choose random inputs to visualize
    indices = [random.randint(0, len(x_test)) for _ in range(n)]
    plt.figure(figsize=(n * 3, 4))
    for vis_i, i in enumerate(indices):
        # Display original
        ax = plt.subplot(3, n, vis_i + 1)
        plt.plot(range(len(x_test[i])), x_test[i], color = "#4D97B2")
        plt.title("Original data")

        # Display encoded
        ax = plt.subplot(3, n, vis_i + 1 + n)
        plt.plot(range(len(encoded_ecg[i])), encoded_ecg[i], color = "#B24D97")
        plt.title("Encoded data")

        # Display reconstructed
        ax = plt.subplot(3, n, vis_i + 1 + 2 * n)
        plt.plot(range(len(decoded_ecg[i])), decoded_ecg[i], color = "#97B24D")
        plt.title("Decoded data")

    plt.tight_layout()
    plt.savefig('./images/results.png', dpi=300, bbox_inches='tight')

    # summarize history for accuracy
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()
    print("Compression={}".format(len(x_test[i])/len(encoded_ecg[i])))
    plt.savefig('accuracy.png'
                , dpi=300
                , bbox_inches='tight')


if __name__ == '__main__':
    main()
