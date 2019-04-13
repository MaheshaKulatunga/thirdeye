import thirdeye
import constants
import cv2
import numpy as np
import os
from keras.utils import to_categorical
import pickle

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

def single_video_test(folder, filename):
    data = []
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


def predict_video():
    videos_to_predict = retrive_data(constants.TEST_SEPARATED_DF_FACES)

    # single_video = single_video_test(constants.TEST_SEPARATED_DF_FACES, '152.mp4')

    providence = pickle.load(open(constants.SAVED_MODELS + 'providence.sav', 'rb'))
    predictions = providence.predict(np.array(videos_to_predict))

    return predictions


if __name__ == "__main__":
    """" USING RAW VIDEOS FIRST"""""
    df_data = retrive_data(constants.TRAIN_SEPARATED_DF_FACES)
    df_labels = [1] * len(df_data)
    print('Found {} Deepfakes'.format(len(df_data)))
    real_data = retrive_data(constants.TRAIN_SEPARATED_REAL_FACES)
    print('Found {} Pristine Videos'.format(len(real_data)))
    real_labels = [0] * len(real_data)

    # Seperate training and test data
    train_x = df_data[:750] + real_data[:750]
    train_y = df_labels[:750] + real_labels[:750]
    train_y = to_categorical(train_y)

    # Remove data from memory
    real_data = []
    df_data =[]

    # Traing model 1 - Providence
    model = thirdeye.providence(np.array(train_x), np.array(train_y), summary=True)

    # predictions = predict_video()
    #
    # for prediction in predictions:
    #     print(round(max(prediction)))
