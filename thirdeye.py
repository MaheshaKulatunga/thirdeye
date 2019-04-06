from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU, ZeroPadding3D
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras import backend as K
import constants
import cv2
import numpy as np
import os
from tensorflow.python.client import device_lib
import pickle

""" Model 1 Basic 3DCNN """
def providence(xtrain, ytrain,summary=False):
    # Input shape
    input_layer = Input((20, 100, 100, 3))

    ## Convolutional layers 1
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

    # Max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer2)

    ## Convolutional layers 2
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)

    # Max pooling to obtain the most imformatic features
    pooling_layer2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer4)

    # Normalize and flatten before feeding it to fully connected classification stage
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)

    # Add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(2, activation='softmax')(dense_layer2)

    # Define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    if summary:
        print(model.summary())

    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])
    history = model.fit(x=xtrain, y=ytrain, batch_size=32, epochs=10, validation_split=0.2, verbose=2)

    # Save the model and history to disk
    filename = constants.SAVED_MODELS + 'providence.sav'
    pickle.dump(model, open(filename, 'wb'))

    his_filename = constants.SAVED_MODELS + 'providence_history.sav'
    pickle.dump(history, open(his_filename, 'wb'))

    return model
