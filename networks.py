from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
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
"""
The networks class is capable of returning any one of our trained networks, given a network name.
"""
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
    xtest, ytest: data to be used for testing
    train: bool to signify if the model is to be trained or loaded
    """
    def load_network(self, name, xtrain=[], ytrain=[], xtest=[], ytest=[], train=False):
        summary=self.summary

        if name == 'odin_v1':
            filters1 = 8
            filters2 = 16
            conv2 = True
            conv4 = False
            nodes_1 = 32
            nodes_2 = 16
            leaky = False

        elif name == 'odin_v2':
            filters1 = 8
            filters2 = 16
            conv2 = True
            conv4 = False
            nodes_1 = 32
            nodes_2 = 16
            leaky = True

        elif name == 'horus':
            filters1 = 16
            filters2 = 16
            conv2 = False
            conv4 = False
            nodes_1 = 32
            nodes_2 = 16
            leaky = False

        elif name == 'providence_v2':
            filters1 = 8
            filters2 = 16
            conv2 = True
            conv4 = True
            nodes_1 = 256
            nodes_2 = 128
            leaky = False

        else:
            filters1 = 8
            filters2 = 16
            conv2 = True
            conv4 = True
            nodes_1 = 2048
            nodes_2 = 512
            leaky = False

        if train:
            # Input shape
            input_layer = Input(xtrain[0].shape)

            ## Convolutional layers 1
            conv_layer1 = Conv3D(filters=filters1, kernel_size=(3, 3, 3), activation='relu')(input_layer)
            # Add 2nd convolution is needed
            if conv2 == True:
                if xtrain[0].shape[0] > 5:
                    conv_layer2 = Conv3D(filters=filters2, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)
                else:
                    conv_layer2 = Conv3D(filters=filters2, kernel_size=(1, 3, 3), activation='relu')(conv_layer1)
            else:
                conv_layer2 = conv_layer1

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

            # Add 4th conv layer if needed
            if conv4 == True:
                # When using less frames, we need to reduce kernal size to fit after previous convolutions
                if xtrain[0].shape[0] > 11:
                    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
                else:
                    conv_layer4 = Conv3D(filters=64, kernel_size=(1, 3, 3), activation='relu')(conv_layer3)
            else:
                conv_layer4 = conv_layer3

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
            dense_layer2 = Dropout(0.4)(dense_layer1)
            if leaky:
                dense_layer3 = LeakyReLU(alpha=5)(dense_layer2)
            else:
                dense_layer3 = Dense(units=nodes_2, activation='relu')(dense_layer2)
            dense_layer4 = Dropout(0.4)(dense_layer3)
            output_layer = Dense(2, activation='softmax')(dense_layer4)

            # Define the model with input layer and output layer
            model = Model(inputs=input_layer, outputs=output_layer)

            if summary:
                print(model.summary())

            model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])
            if len(xtest) > 0 and len(ytest) > 0:
                history = model.fit(x=xtrain, y=ytrain, batch_size=32, epochs=10, validation_data=(xtest, ytest), verbose=2)
            else:
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
                if summary:
                    print(model.summary())
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
