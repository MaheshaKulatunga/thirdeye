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

class Thirdeye:
    """ Initialize Class """
    def __init__(self, pre_p=False, force_t=False, name='providence', evaluate=False, max_f_class=100000, frame_c=3):
        self.PRE_PROCESSING = pre_p
        self.FORCE_TRAIN = force_t
        self.EVALUATE = evaluate
        self.model = None
        self.name = name
        self.title = name.capitalize()
        self.MAX_FOR_CLASS = max_f_class
        self.FRAME_CLIP = frame_c

        if self.PRE_PROCESSING:
            self.preprocess()

        # """" TRAIN MODELS IF NOT ALREADY SAVED """""
        filepath = constants.SAVED_MODELS + self.name + '.sav'
        exists = os.path.isfile(filepath)
        if not exists or self.FORCE_TRAIN:
            self.train()

        self.load()

        if self.EVALUATE:
            self.evaluate()

    """ Preprocess data """
    def preprocess(self):
        preprocessing.handle_train_files()
        preprocessing.handle_test_files()

    """ Train data """
    def train(self):
        print('Training {}'.format(self.title))
        train_x, train_y = self.prepare_rgb_input(self.MAX_FOR_CLASS, self.FRAME_CLIP)
        model = networks.Network(summary=True)

        if self.name == 'providence':
            self.model = model.providence(train_x, train_y, frame_clip=self.FRAME_CLIP)

        if self.name == 'odin':
            self.model = model.odin(train_x, train_y, frame_clip=self.FRAME_CLIP)

        if self.name == 'horus':
            self.model = model.horus(train_x, train_y, frame_clip=self.FRAME_CLIP)

    """ Load saved models """
    def load(self):
        filepath = constants.SAVED_MODELS + self.name + '.sav'
        exists = os.path.isfile(filepath)
        if exists:
            model = networks.Network(summary=True)

            if self.name == 'providence':
                self.model = model.providence(train=False)

            if self.name == 'odin':
                self.model = model.odin(train=False)

            if self.name == 'horus':
                self.model = model.horus(train=False)
        else:
            print('No saved model!')
            exit()

    """ Evaluate models available with seperate data """
    def evaluate(self):
        eval_x, eval_y = self.prepare_rgb_input(self.MAX_FOR_CLASS, self.FRAME_CLIP, test=True)

        history = pickle.load(open(constants.SAVED_MODELS + self.name + '_history.sav', 'rb'))
        print('History of {} loaded'.format(self.title))
        eval = evaluate.Evaluator(self.model)
        eval.plot_accloss_graph(history, self.title)
        eval.predict_test_data(eval_x, eval_y, self.title)

    def classify(self):
        # preprocessing.handle_unknown_files()
        filenames = os.listdir(constants.UNKNOWN_SEP)
        filenames.sort()
        unknown_videos = self.retrive_data(constants.UNKNOWN_SEP)

        classifier = classify.Classifier(self.model)

        for index,video in enumerate(unknown_videos):
            print(filenames[index])
            print(unknown_videos[index])

        print(filenames)


    """ Retrive data from folders"""
    def retrive_data(self, folder, rgb=True, mv_type='mag'):
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

    """ Flip and duplicate videos to increase training set """
    def flip_duplicate(self, data):
        flipped_videos = []
        for video in data:
            new_video = []
            for frame in video:
                new_frame = utilities.flip_img(frame)
                new_video.append(new_frame)
            flipped_videos.append(new_video)

        return flipped_videos

    """ Split videos by defined number of frames """
    def split_frames(self, data, chunk):
        split_fs = []

        for video in data:
            split_fs = split_fs + [video[i:i + chunk] for i in range(0, len(video), chunk)]

        split_fs = [item for item in split_fs if len(item) == chunk]

        return split_fs

    """ Prepare training img data """
    def prepare_rgb_input(self, total_data=1000, frame_clip=-1, test=False):
        if test:
            df_data = self.retrive_data(constants.TEST_SEPARATED_DF_FACES)
        else:
            df_data = self.retrive_data(constants.TRAIN_SEPARATED_DF_FACES)

        # Split further?
        if frame_clip != -1:
            df_data = self.split_frames(df_data, frame_clip)

        if not test:
            # Flip and duplicate
            flipped_df = self.flip_duplicate(df_data)
            df_data = df_data + flipped_df

        df_labels = [1] * len(df_data)
        print('Found {} Deepfakes'.format(len(df_data)))

        if test:
            real_data = self.retrive_data(constants.TEST_SEPARATED_REAL_FACES)
        else:
            real_data = self.retrive_data(constants.TRAIN_SEPARATED_REAL_FACES)


        # Split further?
        if frame_clip != -1:
            real_data = self.split_frames(real_data, frame_clip)
        if not test:
            # Flip and duplicate
            flipped_real = self.flip_duplicate(real_data)
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

    """ Prepare training mc data """
    def prepare_mv_input(self, total_data=1000, frame_clip=-1, rgb=True):
        df_data = retrive_data(constants.TRAIN_MV_DF_FACES, rgb=rgb)

        # # Split further?
        if frame_clip != -1:
            df_data = split_frames(df_data, frame_clip)

        df_labels = [1] * len(df_data)
        print('Found {} Deepfake MVs'.format(len(df_data)))

        real_data = self.retrive_data(constants.TRAIN_MV_REAL_FACES, rgb=rgb)
        # Split further?
        if frame_clip != -1:
            real_data = self.split_frames(real_data, frame_clip)

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
