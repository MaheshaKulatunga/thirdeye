from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Dropout, Reshape, Concatenate
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
import constants

class Network:
    """
    Initialize Class
    -----------------------------------------------------------
    summary: weather or not to print model summary when training or loading
    name: name of the model to be loaded
    """
    def __init__(self, summary=False, name=''):
        self.summary = summary
        self.model = None

        if len(name) > 0:
            self.load_network(name)

    """
    Load Model given name
    -----------------------------------------------------------
    name: name of the model to be loaded
    xtrain, ytrain: data to be used for training
    train: bool to signify if the model is to be trained or loaded
    """
    def load_network(self, name, xtrain=[], ytrain=[], train=False):
        summary=self.summary

        if name == 'odin':
            nodes_1 = 512
            nodes_2 = 256
        elif name == 'horus':
            nodes_1 = 256
            nodes_2 = 128
        else:
            nodes_1 = 2048
            nodes_2 = 512

        if train:
            # Input shape
            input_layer = Input(xtrain[0].shape)

            ## Convolutional layers 1
            conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
            if xtrain[0].shape[0] > 5:
                conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)
            else:
                conv_layer2 = Conv3D(filters=16, kernel_size=(1, 3, 3), activation='relu')(conv_layer1)

            # Max pooling to obtain the most imformatic features
            if xtrain[0].shape[0] > 5:
                pooling_layer1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer2)
            else:
                pooling_layer1 = MaxPooling3D(pool_size=(1, 2, 2))(conv_layer2)

            ## Convolutional layers 2
            if xtrain[0].shape[0] > 8:
                conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
            else:
                conv_layer3 = Conv3D(filters=32, kernel_size=(1, 3, 3), activation='relu')(pooling_layer1)
            # When using less frames, we need to reduce kernal size to fit after previous convolutions
            if xtrain[0].shape[0] > 11:
                conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
            else:
                conv_layer4 = Conv3D(filters=64, kernel_size=(1, 3, 3), activation='relu')(conv_layer3)

            # Max pooling to obtain the most imformatic features
            # When using less frames, we need to reduce kernal size to fit after previous convolutions
            if xtrain[0].shape[0] > 14:
                pooling_layer2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer4)
            else:
                pooling_layer2 = MaxPooling3D(pool_size=(1, 2, 2))(conv_layer4)

            # Normalize and flatten before feeding it to fully connected classification stage
            pooling_layer2 = BatchNormalization()(pooling_layer2)
            flatten_layer = Flatten()(pooling_layer2)

            # Add dropouts to avoid overfitting / perform regularization
            dense_layer1 = Dense(units=nodes_1, activation='relu')(flatten_layer)
            dense_layer1 = Dropout(0.4)(dense_layer1)
            dense_layer2 = Dense(units=nodes_2, activation='relu')(dense_layer1)
            dense_layer2 = Dropout(0.4)(dense_layer2)
            output_layer = Dense(2, activation='softmax')(dense_layer2)

            # Define the model with input layer and output layer
            model = Model(inputs=input_layer, outputs=output_layer)

            if summary:
                print(model.summary())

            model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])
            history = model.fit(x=xtrain, y=ytrain, batch_size=32, epochs=10, validation_split=0.2, verbose=2)

            # Save the model and history to disk
            filename = constants.SAVED_MODELS + name + '.sav'
            pickle.dump(model, open(filename, 'wb'))

            his_filename = constants.SAVED_MODELS + name + '_history.sav'
            pickle.dump(history, open(his_filename, 'wb'))
        else:
            providence_filepath = constants.SAVED_MODELS + name + '.sav'
            exists = os.path.isfile(providence_filepath)
            if exists:
                model = pickle.load(open(constants.SAVED_MODELS + name + '.sav', 'rb'))
                print('{} is ready.'.format(name.capitalize()))
            else:
                prin('No saved model detected!')
        self.model = model

    """
    Set Model
    -----------------------------------------------------------
    Change the current active model
    name: name of the model to be loaded
    xtrain, ytrain: data to be used for training
    train: bool to signify if the model is to be trained or loaded
    """
    def set_model(self, name, xtrain=[], ytrain=[], train=False):
        self.load_network(name, xtrain=[], ytrain=[], train=False)

    """
    Get Model
    -----------------------------------------------------------
    Returns the current active model object
    """
    def get_model(self):
        return self.model
