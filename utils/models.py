import pandas as pd
from keras.models import Model, load_model
from keras import regularizers, optimizers

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import SGD, RMSprop, Adam

def autoencoder_(X,input_size, hidden_size=64, code_size=50):
    input_data = Input(shape=(310,))
    encoded1 = Dense(310,input_shape=(310,))(input_data)
    encoded2 = Activation('relu')(encoded1)
    encoded3 = BatchNormalization()(encoded2)
    code = Dense(code_size)(encoded3)
    decoded1 = Dense(310)(code)
    decoded2= Activation('relu')(decoded1)
    decoded3 =BatchNormalization()(decoded2)
    decoded4 = Dense(310, activation='sigmoid')(decoded3)

    autoencoder = Model(inputs=input_data, outputs=decoded4)
    autoencoder.compile(optimizer = RMSprop(), loss = 'binary_crossentropy', metrics=['mae'])
    history = autoencoder.fit(X, X, epochs=10, batch_size=1024)

    reduce_feature = Model(inputs = input_data, outputs = code)
    encoded_input = Input(shape = (input_size, ))

    encoded_train = pd.DataFrame(reduce_feature.predict(X))
    return encoded_train

def nn_model():
    network = Sequential()
    network.add(Dense(128, input_shape=(310,), kernel_regularizer=regularizers.l2(0.0001)))
    network.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    network.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    network.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    network.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    network.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)))
    network.compile(optimizer=optimizers.Adam(lr=0.001), loss='mse', metrics=['mae'])
    return network
