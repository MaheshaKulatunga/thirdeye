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
import sys

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
            self.perform_preprocessing()

        if self.FORCE_TRAIN:
            self.train()
        else:
            self.load()

        if self.EVALUATE and (self.model is not None):
            self.evaluate()

    """ Preprocess data """
    def perform_preprocessing(self):
        pre_p = preprocessing.Preprocessor()
        try:
            pre_p.preprocess(1)
            pre_p.preprocess(2)
        except:
            print("Oops!",sys.exc_info()[0],"occured.")

    """ Train data """
    def train(self):
        try:
            print('Training {}'.format(self.title))

            model = networks.Network(summary=True)

            if self.network == 'providence':
                train_x, train_y = self.prepare_rgb_input(self.MAX_FOR_CLASS, self.FRAME_CLIP)

                if len(train_x) == 0 or len(train_y) == 0:
                    print('No training data!')
                    exit()
                model.load_network('providence', train_x, train_y, train=True)
                self.model = model.get_model()

            elif self.network == 'odin':
                train_x, train_y = self.prepare_rgb_input(self.MAX_FOR_CLASS, self.FRAME_CLIP)

                if len(train_x) == 0 or len(train_y) == 0:
                    print('No training data!')
                    exit()
                model.load_network('odin', train_x, train_y, train=True)
                self.model = model.get_model()

            elif self.network == 'horus':
                train_x, train_y = self.prepare_rgb_input(self.MAX_FOR_CLASS, self.FRAME_CLIP)

                if len(train_x) == 0 or len(train_y) == 0:
                    print('No training data!')
                    exit()
                model.load_network('horus', train_x, train_y, train=True)
                self.model = model.get_model()

            else:
                print('Invalid network {}, reverting to Default'.format(self.title))
                self.set_network('providence')
        except:
            print("Oops!",sys.exc_info()[0],"occured.")

    """ Load saved models """
    def load(self):
        filepath = constants.SAVED_MODELS + self.network + '.sav'
        exists = os.path.isfile(filepath)
        if exists:
            model = networks.Network(summary=True)

            if self.network == 'providence':
                model.load_network('providence')
                self.model = model.get_model()

            if self.network == 'odin':
                model.load_network('odin')
                self.model = model.get_model()

            if self.network == 'horus':
                model.load_network('horus')
                self.model = model.get_model()
        else:
            print('No saved network {}! Attempting to train it.'.format(self.title))
            self.train()

    """ Evaluate models available with seperate data """
    def evaluate(self):
        try:
            eval_x, eval_y = self.prepare_rgb_input(self.MAX_FOR_CLASS, self.FRAME_CLIP, test=True)

            history = pickle.load(open(constants.SAVED_MODELS + self.network + '_history.sav', 'rb'))
            print('History of {} loaded'.format(self.title))

            eval = evaluate.Evaluator(self.model)
            eval.plot_accloss_graph(history, self.title)
            eval.predict_test_data(eval_x, eval_y, self.title)
        except:
            print("Oops!",sys.exc_info()[0],"occured.")

    """ Classify an unknown video """
    def classify(self):
        try:
            if len(os.listdir(constants.UNKNOWN_RAW)) > 0:
                pre_p = preprocessing.Preprocessor()
                pre_p.preprocess(3)

            classifier = classify.Classifier(self.model, constants.UNKNOWN_SEP, self.FRAME_CLIP)
            predictions = classifier.classify_videos()
            for index, video in enumerate(predictions.keys()):
                print('========== Video {} =========='.format(video))
                print('Real: {}%, Deepfake: {}%'.format(round(predictions[video]['Real']*100, 2), round(predictions[video]['Deepfake']*100, 2)))
                if predictions[video]['Real'] > predictions[video]['Deepfake']:
                    label = 'Real'
                else:
                    label = 'Deepfake'

                print('{}: {} \n'.format(video, label))

            # utilities.clear_folder(constants.UNKNOWN_SEP)
            return predictions
        except:
            print("Oops!",sys.exc_info()[0],"occured.")

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

    """ Prepare training mc data ----------------------- DEPRECIATED -----------------------------"""
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

    """ Switch networks """
    def set_network(self, network):
        self.network = network
        self.title = network.capitalize()
        try:
            self.load()
        except:
            print("Oops!",sys.exc_info()[0],"occured.")

    """ Set frame clip """
    def set_frame_clip(self, frame_clip):
        self.FRAME_CLIP = frame_clip

    """ Set max for class """
    def set_max_for_class(self, max_for_class):
        self.MAX_FOR_CLASS = max_for_class

    """ Get max for class """
    def get_max_for_class(self):
        return self.MAX_FOR_CLASS

    """ Get frame clip """
    def get_frame_clip(self):
        return self.FRAME_CLIP

    """ Get Network """
    def get_network(self):
        return {self.title: self.model}
