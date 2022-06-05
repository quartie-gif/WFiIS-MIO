import keras
import keras.layers
import numpy as np
from keras.datasets import mnist


class EcgAutoencoder(keras.Model):
    """
    Klasa daje dostęp do trzech MLP: encoder, decoder i autoencoder
    Po konstrukcji instancji są już skompilowane, trzeba tylko włączyć
    fitowanie na interesujących nas danych.
    """
    def __init__(self,
                 input_size,
                 layer_sizes=None,
                 activations=None,
                 encoder_output_index=None,
                 loss='binary_crossentropy'):
        """
        :param input_size: Rozmiar danych wejściowych
        :param layer_sizes: Lista rozmiarów warstw ukrytych (tj. nie licząc wejścia i wyjścia, które mają rozmiar input_size)
        :param activations: Lista aktywacji warstw ukrytych (warstwa wyjściowa ma aktywację sigmoid)
        :param encoder_output_index: Indeks warstwy, która ma być wyjściem dekodera. Jeśli nie podano, będzie równy indeksowi najmniejszej warstwy.
        :param loss: Wiadomo
        """
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [150, 70, 30, 70, 150]
        if activations is None:
            activations = ['relu' for _ in range(layer_sizes)]
        if encoder_output_index is None:
            encoder_output_index = next(i for i, v in enumerate(layer_sizes) if v == min(layer_sizes))
        layer_sizes.append(input_size)
        activations.append('sigmoid')
        self.input_l = keras.Input(shape=(input_size,))
        self.dense_layers = [keras.layers.Dense(size, activation)
                             for size, activation in zip(layer_sizes, activations)]
        # Build models
        self.encoder_layers = self.dense_layers[:encoder_output_index+1]
        self.decoder_layers = self.dense_layers[encoder_output_index+1:]
        tensor = self.input_l
        for i, layer in enumerate(self.dense_layers):
            tensor = layer(tensor)
            if i == encoder_output_index:
                encoded_tensor = tensor
        self.autoencoder = keras.Model(self.input_l, tensor)
        self.encoder = keras.Model(self.input_l, encoded_tensor)
        encoded_input = keras.Input(shape=(layer_sizes[encoder_output_index],))
        tensor = encoded_input
        for layer in self.decoder_layers:
            tensor = layer(tensor)
        self.decoder = keras.Model(encoded_input, tensor)
        self.autoencoder.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_autoencoder(self):
        return self.autoencoder

    def call(self, inputs, training=None, mask=None):
        return self.decoder_layers[-1]
