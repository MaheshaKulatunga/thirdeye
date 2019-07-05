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

def split_frames(data, chunk):
    split_fs = []

    for video in data:
        split_fs = split_fs + [video[i:i + chunk] for i in range(0, len(video), chunk)]

    split_fs = [item for item in split_fs if len(item) == chunk]

    return split_fs

def pic_2_vid(files, pathIn, pathOut):

    videos = split_frames(files.values(), 20)

    for index, file in enumerate(videos):
        frame_array = []
        for i in range(len(file)):
            filename=pathIn + file[i]
            #reading each files
            img = cv2.imread(filename)
            img = cv2.resize(img,(100,100))
            height, width, layers = img.shape
            size = (width,height)

            #inserting the frames into an image array
            frame_array.append(img)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = pathOut.format(index)
        print(output_path)
        video = cv2.VideoWriter(output_path, fourcc, 20, size)

        # Appending the images to the video one by one
        for image in frame_array:
            video.write(image)

        # Deallocating memories taken for window creation
        cv2.destroyAllWindows()
        video.release()  # releasing the video generated

def vid_2_pic(files, pathIn, pathOut):

    for index, video in enumerate(files):
        filename=pathIn + video
        vidcap = cv2.VideoCapture(filename)
        success,image = vidcap.read()
        count = 0
        while success:
            output_path = pathOut.format(index, count)
            image = cv2.resize(image, (256,256))
            cv2.imwrite(output_path, image)     # save frame as JPEG file
            success,image = vidcap.read()
            count += 1

if __name__ == "__main__":
    # # PIC 2 VID
    # files = get_filenames('/run/media/u1856817/KINGSTON/MesoNetData/df/')
    # grouped = group_by_video(files)
    # # For each group, read in the files as frames and split out video
    # pic_2_vid(grouped, '/run/media/u1856817/KINGSTON/MesoNetData/df/', '/run/media/u1856817/KINGSTON/MesoNetData/df_vids/{}.mp4')
    #
    # files = get_filenames('/run/media/u1856817/KINGSTON/MesoNetData/real/')
    # grouped = group_by_video(files)
    # pic_2_vid(grouped, '/run/media/u1856817/KINGSTON/MesoNetData/real/', '/run/media/u1856817/KINGSTON/MesoNetData/real_vids/{}.mp4')
    #
    # files = get_filenames('/run/media/u1856817/KINGSTON/MesoNetData/TRAIN/df/')
    # grouped = group_by_video(files)
    # # For each group, read in the files as frames and split out video
    # pic_2_vid(grouped, '/run/media/u1856817/KINGSTON/MesoNetData/TRAIN/df/', '/run/media/u1856817/KINGSTON/MesoNetData/TRAIN/df_vids/{}.mp4')
    #
    # files = get_filenames('/run/media/u1856817/KINGSTON/MesoNetData/TRAIN/real/')
    # grouped = group_by_video(files)
    # pic_2_vid(grouped, '/run/media/u1856817/KINGSTON/MesoNetData/TRAIN/real/', '/run/media/u1856817/KINGSTON/MesoNetData/TRAIN/real_vids/{}.mp4')

    #VID 2 PIC
    files = get_filenames('/run/media/u1856817/KINGSTON/DF_SEP/')
    vid_2_pic(files, '/run/media/u1856817/KINGSTON/DF_SEP/', '/run/media/u1856817/KINGSTON/DF_SEP_IMGS/{}_{}.jpg')

    files = get_filenames('/run/media/u1856817/KINGSTON/REAL_SEP/')
    vid_2_pic(files, '/run/media/u1856817/KINGSTON/REAL_SEP/', '/run/media/u1856817/KINGSTON/REAL_SEP_IMGS/{}_{}.jpg')
