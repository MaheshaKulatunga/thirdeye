import thirdeye
import constants
import cv2
import numpy as np
import os
from keras.utils import to_categorical

""" Retrive data from folders"""
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
    df_data = retrive_data(constants.TRAIN_SEPARATED_DF_FACES)
    df_labels = [1] * len(df_data)
    print('Found {} Deepfakes'.format(len(df_data)))
    real_data = retrive_data(constants.TRAIN_SEPARATED_REAL_FACES)
    print('Found {} Pristine Videos'.format(len(real_data)))
    real_labels = [0] * len(real_data)

    # Seperate training and test data
    train_x = df_data[:500] + real_data[:500]
    train_y = df_labels[:500] + real_labels[:500]
    train_y = to_categorical(train_y)

    # Remove data from memory
    real_data = []
    df_data =[]

    # Traing model 1
    model = thirdeye.providence(np.array(train_x), np.array(train_y), summary=True)
    
