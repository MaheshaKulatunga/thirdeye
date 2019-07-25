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
    def __init__(self, pre_p=False, force_t=False, network='providence', evaluate=False, max_for_class=100000, frame_clip=3):
        self.PRE_PROCESSING = pre_p
        self.FORCE_TRAIN = force_t
        self.EVALUATE = evaluate
        self.model = None
        self.network = network
        self.title = network.capitalize()
        self.MAX_FOR_CLASS = max_for_class
        self.FRAME_CLIP = frame_clip

        if self.PRE_PROCESSING:
            self.preprocess()

        if self.FORCE_TRAIN:
            self.train()

        self.load()

        if self.EVALUATE and (self.model is not None):
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

        if self.network == 'providence':
            self.model = model.load_network('providence', train_x, train_y, frame_clip=self.FRAME_CLIP, train=True)

        if self.network == 'odin':
            self.model = model.load_network('odin', train_x, train_y, frame_clip=self.FRAME_CLIP, train=True)

        if self.network == 'horus':
            self.model = model.load_network('horus', train_x, train_y, frame_clip=self.FRAME_CLIP, train=True)

    """ Load saved models """
    def load(self):
        filepath = constants.SAVED_MODELS + self.network + '.sav'
        exists = os.path.isfile(filepath)
        if exists:
            model = networks.Network(summary=True)

            if self.network == 'providence':
                self.model = model.load_network('providence')

            if self.network == 'odin':
                self.model = model.load_network('odin')

            if self.network == 'horus':
                self.model = model.load_network('horus')
        else:
            print('No saved model!')

    """ Evaluate models available with seperate data """
    def evaluate(self):
        eval_x, eval_y = self.prepare_rgb_input(self.MAX_FOR_CLASS, self.FRAME_CLIP, test=True)

        history = pickle.load(open(constants.SAVED_MODELS + self.network + '_history.sav', 'rb'))
        print('History of {} loaded'.format(self.title))

        eval = evaluate.Evaluator(self.model)
        eval.plot_accloss_graph(history, self.title)
        eval.predict_test_data(eval_x, eval_y, self.title)

    """ Classify an unknown video """
    def classify(self):
        if len(os.listdir(constants.UNKNOWN_RAW)) > 0:
            preprocessing.handle_unknown_files()

        classifier = classify.Classifier(self.model, constants.UNKNOWN_SEP, self.FRAME_CLIP)
        classifier.classify_videos()
        utilities.clear_folder(constants.UNKNOWN_SEP)

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

    """ Prepare training img data """
    def prepare_rgb_input(self, total_data=1000, frame_clip=-1, test=False):
        if test:
            df_data = utilities.retrieve_data(constants.TEST_SEPARATED_DF_FACES)
        else:
            df_data = utilities.retrieve_data(constants.TRAIN_SEPARATED_DF_FACES)

        # Split further?
        if frame_clip != -1:
            df_data = utilities.split_frames(df_data, frame_clip)

        if not test:
            # Flip and duplicate
            flipped_df = self.flip_duplicate(df_data)
            df_data = df_data + flipped_df

        df_labels = [1] * len(df_data)
        print('Found {} Deepfakes'.format(len(df_data)))

        if test:
            real_data = utilities.retrieve_data(constants.TEST_SEPARATED_REAL_FACES)
        else:
            real_data = utilities.retrieve_data(constants.TRAIN_SEPARATED_REAL_FACES)


        # Split further?
        if frame_clip != -1:
            real_data = utilities.split_frames(real_data, frame_clip)
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
        df_data = utilities.retrieve_data(constants.TRAIN_MV_DF_FACES, rgb=rgb)

        # # Split further?
        if frame_clip != -1:
            df_data = utilities.split_frames(df_data, frame_clip)

        df_labels = [1] * len(df_data)
        print('Found {} Deepfake MVs'.format(len(df_data)))

        real_data = utilities.retrieve_data(constants.TRAIN_MV_REAL_FACES, rgb=rgb)
        # Split further?
        if frame_clip != -1:
            real_data = utilities.split_frames(real_data, frame_clip)

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

    def set_network(self, network):
        self.network = network
        self.title = network.capitalize()
        self.load()

    def set_frame_clip(self, frame_clip):
        self.FRAME_CLIP = frame_clip

    def set_max_for_class(self, max_for_class):
        self.MAX_FOR_CLASS = max_for_class
