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

def get_model(xtrain, ytrain,summary=False):
    """ Return the Keras model of the network
    """
    # model = Sequential()
    # # 1st layer group
    # model.add(Conv3D(64, 3, 3, 3, activation='relu',
    #                         border_mode='same', name='conv1',
    #                         subsample=(1, 1, 1),
    #                         input_shape=(20, 100, 100, 3)))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
    #                        border_mode='valid', name='pool1', dim_ordering="th"))
    # 2nd layer group
    # model.add(Conv3D(64, 3, 3, 3, activation='relu',
    #                         border_mode='same', name='conv2',
    #                         subsample=(1, 1, 1)))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
    #                        border_mode='valid', name='pool2', dim_ordering="th"))
    # # 3rd layer group
    # model.add(Conv3D(256, 3, 3, 3, activation='relu',
    #                         border_mode='same', name='conv3a',
    #                         subsample=(1, 1, 1)))
    # model.add(Conv3D(256, 3, 3, 3, activation='relu',
    #                         border_mode='same', name='conv3b',
    #                         subsample=(1, 1, 1)))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
    #                        border_mode='valid', name='pool3', dim_ordering="th"))
    # # 4th layer group
    # model.add(Conv3D(512, 3, 3, 3, activation='relu',
    #                         border_mode='same', name='conv4a',
    #                         subsample=(1, 1, 1)))
    # model.add(Conv3D(512, 3, 3, 3, activation='relu',
    #                         border_mode='same', name='conv4b',
    #                         subsample=(1, 1, 1)))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
    #                        border_mode='valid', name='pool4', dim_ordering="th"))
    # # 5th layer group
    # model.add(Conv3D(512, 3, 3, 3, activation='relu',
    #                         border_mode='same', name='conv5a',
    #                         subsample=(1, 1, 1)))
    # model.add(Conv3D(512, 3, 3, 3, activation='relu',
    #                         border_mode='same', name='conv5b',
    #                         subsample=(1, 1, 1)))
    # model.add(ZeroPadding3D(padding=(0, 1, 1)))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
    #                        border_mode='valid', name='pool5', dim_ordering="th"))
    # model.add(Flatten())
    # # FC layers group
    # # model.add(Dense(248, activation='relu', name='fc6'))
    # model.add(Dropout(.5))
    # model.add(Dense(128, activation='relu', name='fc7'))
    # model.add(Dropout(.5))
    # model.add(Dense(2, activation='softmax', name='fc8'))

    input_layer = Input((20, 100, 100, 3))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

    ## add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer2)

    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
    pooling_layer2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer4)

    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)

    ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
    ## add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(2, activation='softmax')(dense_layer2)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    if summary:
        print(model.summary())
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])
    model.fit(x=xtrain, y=ytrain, batch_size=32, epochs=5, validation_split=0.2, verbose=2)

    return model



def retrive_data(folder):
    data = []
    for index, filename in enumerate(os.listdir(folder)):
        cap = cv2.VideoCapture(folder + filename)

        vid = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            vid.append(img)
        vid = np.array(vid, dtype=np.float32)
        data.append(vid)
    return data

if __name__ == "__main__":
    # print(device_lib.list_local_devices())
    # print(K.tensorflow_backend._get_available_gpus())
    df_data = retrive_data(constants.TRAIN_SEPARATED_DF_FACES)
    df_labels = [1] * len(df_data)
    real_data = retrive_data(constants.TRAIN_SEPARATED_REAL_FACES)
    real_labels = [0] * len(real_data)
    print(len(real_data))
    print(len(df_data))
    train_x = df_data[:500] + real_data[:500]
    train_y = df_labels[:500] + real_labels[:500]
    train_y = to_categorical(train_y)

    real_data = []
    df_data =[]
    model = get_model(np.array(train_x), np.array(train_y), summary=True)
