# import thirdeye
# import preprocessing
# import constants
# import utilities
# import evaluate
import cv2
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.utils import shuffle

# Get filenames
def get_filenames(folder):
    data = []
    sorted_folder = os.listdir(folder)
    sorted_folder.sort()

    return sorted_folder



# Group by number before _ and add to a data list
def group_by_video(filenames):
    # Take first file, find all files with that number before _, concatinate them as one sample
    grouped_files = {}
    while len(filenames) > 0:
        frame_group = []

        first_frame = filenames[0]
        filenames.remove(first_frame)
        frame_group += [first_frame]

        ff_id = first_frame.split('_')[0]
        temp_list = []
        for file in filenames:
            video_id = file.split('_')[0]
            if video_id == ff_id:
                frame_group = frame_group + [file]
                temp_list.append(file)
        filenames = list(set(filenames) - set(temp_list))
        grouped_files.update({ff_id: frame_group})

    return grouped_files

def pic_2_vid(files):

    files = files['170']
    frame_array = []

    pathIn = '/run/media/u1856817/KINGSTON/MesoNetData/df/'

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        img = cv2.resize(img,(250,250))
        height, width, layers = img.shape
        size = (width,height)

        #inserting the frames into an image array
        frame_array.append(img)

    video = cv2.VideoWriter('1.mp4', 4, 20, size)

    # Appending the images to the video one by one
    for image in frame_array:
        video.write(image)

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


if __name__ == "__main__":
    files = get_filenames('/run/media/u1856817/KINGSTON/MesoNetData/df/')
    grouped = group_by_video(files)
    # For each group, read in the files as frames and split out video
    pic_2_vid(grouped)
