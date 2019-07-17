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

        pred_b = averaged_array
        # corr = 0
        for index, video in enumerate(self.unknown_videos):
            if pred_b[index][0] > pred_b[index][1]:
                label = 'Real'
            else:
                label = 'Deepfake'

            print('{}: {}'.format(self.filenames[index], label))
