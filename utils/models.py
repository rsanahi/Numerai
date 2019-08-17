from keras.models import Model, load_model
from keras import models
from keras import layers
from keras import regularizers, optimizers

def autoencoder_(input_size, hidden_size=64, code_size=32):
    input_data = layers.Input(shape=(input_size,))
    hidden_1 = layers.Dense(hidden_size, activation='relu')(input_data)
    code = layers.Dense(code_size, activation='relu')(hidden_1)
    hidden_2 = layers.Dense(hidden_size, activation='relu')(code)
    output_data = layers.Dense(input_size, activation='sigmoid')(hidden_2)
    autoencoder = Model(input_data, output_data)
    autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return autoencoder

def nn_model():
    network = models.Sequential()
    network.add(layers.Dense(128, input_shape=(310,), kernel_regularizer=regularizers.l2(0.0001)))
    network.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    network.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    network.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    network.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    network.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    network.compile(optimizer=optimizers.Adam(lr=0.001), loss='mse', metrics=['mae'])
    return network