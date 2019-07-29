import constants
import utilities
import pickle
import os

class Classifier:

    def __init__(self, model, folder, frames):
        self.model = model
        self.filenames = os.listdir(folder)
        self.filenames.sort()
        self.unknown_videos = utilities.retrieve_data(folder)
        self.unknown_clips = utilities.split_frames(self.unknown_videos, frames)

    """ Classify unknown videos """
    def classify_videos(self):
        pred = self.model.predict([self.unknown_clips])

        n_averaged_elements = 6
        averaged_array = []
        a = pred
        for i in range(0, len(a), n_averaged_elements):
            slice_from_index = i
            slice_to_index = slice_from_index + n_averaged_elements
            slice = a[slice_from_index:slice_to_index]
            class_1_avg = 0
            class_2_avg = 0
            for val in slice:
               class_1_avg += val[0]
               class_2_avg += val[1]

            class_1_avg = class_1_avg/ n_averaged_elements
            class_2_avg = class_2_avg/ n_averaged_elements

            averaged_array.append([class_1_avg, class_2_avg])

        pred_b = {}

        # corr = 0
        for index, video in enumerate(self.unknown_videos):
            pred_b.update({self.filenames[index]: {'Real': averaged_array[index][0], 'Deepfake': averaged_array[index][1]}})

        return pred_b

    """ Set frames """
    def set_frames(self, frames):
        self.unknown_clips = utilities.split_frames(self.unknown_videos, frames)

    """ Set folder """
    def set_folder(self, folder):
        self.filenames = os.listdir(folder)
        self.filenames.sort()
        self.unknown_videos = utilities.retrieve_data(folder)
        self.unknown_clips = utilities.split_frames(self.unknown_videos, self.frames)

    """ Set model """
    def set_model(self, model):
        self.model = model

    """ Get frames """
    def get_frames(self):
        return self.frames

    """ Get folder """
    def get_folder(self):
        return self.folder

    """ Get model """
    def get_model(self):
        return self.model
