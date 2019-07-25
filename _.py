import networks
import classify
import preprocessing
import constants
import utilities
import evaluate
import cv2
import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
import pickle
from sklearn.utils import shuffle
import json

""" Retrive data from folders"""
def retrive_data(folder, rgb=True, mv_type='mag'):
    data = []
    sorted_folder = os.listdir(folder)
    sorted_folder.sort()
    if rgb:
        print("Retriving videos from file")
        for index, filename in enumerate(sorted_folder):
            cap = cv2.VideoCapture(folder + filename)

            vid = []
            while True:
                ret, img = cap.read()
                if not ret:
                    break
                vid.append(img)
            vid = np.array(vid, dtype=np.float32)
            data.append(vid)
    else:
        print("Retriving motion vectors from file")
        for index, filename in enumerate(sorted_folder):
            with open(folder + filename, 'r') as f:
                mv = json.load(f)
            mv_arr = np.array(mv[mv_type], dtype=np.float32)
            data.append(mv_arr)

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

def flip_duplicate(data):
    flipped_videos = []
    for video in data:
        new_video = []
        for frame in video:
            new_frame = utilities.flip_img(frame)
            new_video.append(new_frame)
        flipped_videos.append(new_video)

    return flipped_videos

def split_frames(data, chunk):
    split_fs = []

    for video in data:
        split_fs = split_fs + [video[i:i + chunk] for i in range(0, len(video), chunk)]

    split_fs = [item for item in split_fs if len(item) == chunk]

    return split_fs

def prepare_training_img_data(total_data=1000, frame_clip=-1, test=False):
    if test:
        df_data = retrive_data(constants.TEST_SEPARATED_DF_FACES)
    else:
        df_data = retrive_data(constants.TRAIN_SEPARATED_DF_FACES)

    # Split further?
    if frame_clip != -1:
        df_data = split_frames(df_data, frame_clip)

    if not test:
        # Flip and duplicate
        flipped_df = flip_duplicate(df_data)
        df_data = df_data + flipped_df

    df_labels = [1] * len(df_data)
    print('Found {} Deepfakes'.format(len(df_data)))

    if test:
        real_data = retrive_data(constants.TEST_SEPARATED_REAL_FACES)
    else:
        real_data = retrive_data(constants.TRAIN_SEPARATED_REAL_FACES)


    # Split further?
    if frame_clip != -1:
        real_data = split_frames(real_data, frame_clip)
    if not test:
        # Flip and duplicate
        flipped_real = flip_duplicate(real_data)
        real_data = real_data + flipped_real

    real_labels = [0] * len(real_data)
    print('Found {} Pristine Videos'.format(len(real_data)))

    train_x = df_data[:total_data] + real_data[:total_data]
    train_y = df_labels[:total_data] + real_labels[:total_data]
    data = {'Videos': train_x, 'Labels':train_y}

    # Create DataFrame to shuffle
    data_frame = pd.DataFrame(data)
    data_frame = shuffle(data_frame, random_state=42)

    # Remove data from memory
    real_data = []
    df_data =[]

    return np.array(list(data_frame['Videos'].values)), np.array(to_categorical(list(data_frame['Labels'])))

def prepare_training_mv_data(total_data=1000, frame_clip=-1, rgb=True):
    df_data = retrive_data(constants.TRAIN_MV_DF_FACES, rgb=rgb)

    # # Split further?
    if frame_clip != -1:
        df_data = split_frames(df_data, frame_clip)

    df_labels = [1] * len(df_data)
    print('Found {} Deepfake MVs'.format(len(df_data)))

    real_data = retrive_data(constants.TRAIN_MV_REAL_FACES, rgb=rgb)
    # Split further?
    if frame_clip != -1:
        real_data = split_frames(real_data, frame_clip)

    real_labels = [0] * len(real_data)
    print('Found {} Pristine MVs'.format(len(real_data)))

    train_x = df_data[:total_data] + real_data[:total_data]
    train_y = df_labels[:total_data] + real_labels[:total_data]
    data = {'MVs': train_x, 'Labels':train_y}

    # Create DataFrame to shuffle
    data_frame = pd.DataFrame(data)
    data_frame = shuffle(data_frame, random_state=42)

    # Remove data from memory
    real_data = []
    df_data =[]

    return np.array(list(data_frame['MVs'].values)), np.array(to_categorical(list(data_frame['Labels'])))

if __name__ == "__main__":

    """ THIRDEYE PARAMETERS """
    # Carry out preprocessing?
    PRE_PROCESSING = False
    # Clip frames below 20?
    FRAME_CLIP = 3
    # Maximum videos per class
    MAX_FOR_CLASS = 100000
    # Force retaining of models?
    FORCE_TRAIN = True
    # Evaluate models?
    EVALUATE = False
    # Activate models
    PROVIDENCE = True
    ODIN = True
    HORUS = True

    """ PREPROCESSING """
    if PRE_PROCESSING:
        preprocessing.handle_train_files()
        preprocessing.handle_test_files()

    """" TRAIN MODELS IF NOT ALREADY SAVED """""
    if PROVIDENCE:

        # Traing model 1 - Providence
        providence_filepath = constants.SAVED_MODELS + 'providence.sav'
        exists = os.path.isfile(providence_filepath)
        if not exists or FORCE_TRAIN:
            print('Training Providence.')
            train_x, train_y = prepare_training_img_data(MAX_FOR_CLASS, FRAME_CLIP)
            model = networks.providence(train_x, train_y, summary=True, frame_clip=FRAME_CLIP)
        else:
            print('Providence is ready.')

    if ODIN and not EVALUATE:

        # Traing model 2 - Odin
        providence_filepath = constants.SAVED_MODELS + 'odin.sav'
        exists = os.path.isfile(providence_filepath)
        if not exists or FORCE_TRAIN:
            print('Training Odin')
            train_x, train_y = prepare_training_img_data(MAX_FOR_CLASS, FRAME_CLIP)
            model = networks.odin(train_x, train_y, summary=True, frame_clip=FRAME_CLIP)
        else:
            print('Odin is ready.')

    if HORUS and not EVALUATE:

        # Traing model 2 - Odin
        providence_filepath = constants.SAVED_MODELS + 'horus.sav'
        exists = os.path.isfile(providence_filepath)
        if not exists or FORCE_TRAIN:
            print('Training Horus')
            train_x, train_y = prepare_training_img_data(MAX_FOR_CLASS, FRAME_CLIP)
            model = networks.odin(train_x, train_y, summary=True, frame_clip=FRAME_CLIP)
        else:
            print('Horus is ready.')

    """ EVALUATE MODEL """
    if EVALUATE and PROVIDENCE:
        print('Evaluating model')

        providence_filepath = constants.SAVED_MODELS + 'providence.sav'
        exists = os.path.isfile(providence_filepath)
        if exists:
            eval_x, eval_y = prepare_training_img_data(MAX_FOR_CLASS, FRAME_CLIP, test=True)
            providence_model = pickle.load(open(constants.SAVED_MODELS + 'providence.sav', 'rb'))
            evaluate.predict_test_data(providence_model, eval_x, eval_y, 'Providence')
        else:
            print('No model saved to evaluate; please train a model first')

        # providence_history = pickle.load(open(constants.SAVED_MODELS + 'providence_history.sav', 'rb'))
        # evaluate.plot_accloss_graph(providence_history, 'Providence')

    if EVALUATE and ODIN:
        print('Evaluating model')
        odin_history = pickle.load(open(constants.SAVED_MODELS + 'odin_history.sav', 'rb'))
        evaluate.plot_accloss_graph(odin_history, 'Odin')
 # input_movie = utilities.init_video(folder + file_name)
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     mult_frame = []
#     while True:
#         # Grab a single frame of video
#         ret, frame = input_movie.read()
#         frame_number += 1
#
#         # Quit when the input video file ends
#         if not ret:
#             break
#
#         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#         rgb_frame = frame[:, :, ::-1]
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#         for (x, y, w, h) in faces:
#
#             frame = frame[y:y+h,x:x+w]
#
#             try:
#                 frame = cv2.resize(frame, (box_size, box_size), interpolation=cv2.INTER_LINEAR)
#                 frame_list = frame_list + [frame]
#
#             except Exception as e:
#                 print(str(e))
#         count += 1
#         mult_frame.append(frame_list)
#
#     # Write the resulting frames to the output video file
#     for frame_list in mult_frame:
#         if len(frame_list) == frames:
#             output_movie = cv2.VideoWriter(output_folder + file_name, fourcc, length, (box_size, box_size))
#
#             for f in range(frames):
#                 print("Writing frame {} / {}".format(f+1, length))
#                 output_movie.write(frame_list[f])
#         else:
#             if len(frame_list) >= (frames * 0.75):
#                 output_movie = cv2.VideoWriter(output_folder + file_name, fourcc, length, (box_size, box_size))
#
#                 print('Duplicating frames for video {}'.format(file_name))
#                 frame_list = frame_list + [frame_list[0]] * (frames - len(frame_list))
#                 for f in range(frames):
#                     print("Writing frame {} / {}".format(f+1, length))
#                     output_movie.write(frame_list[f])
#             else:
#                 print('Discarding invalid video {}'.format(file_name))
#
#         # All done!
#         input_movie.release()
#         cv2.destroyAllWindows()
