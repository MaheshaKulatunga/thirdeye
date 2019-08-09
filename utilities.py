import cv2
import os
import numpy as np
import csv
import ast
"""
Contains reusable code and constants that are used throughout the application.
"""
"""
Initialize video
-----------------------------------------------------------
Initialises a video as CV2 object
"""
def init_video(filepath):
    vid = cv2.VideoCapture(filepath)
    return vid

"""
Delete contents of a folder
-----------------------------------------------------------
folder: folder to delete
"""
def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    print('{} cleared'.format(folder))

"""
Get number of frames in videos
-----------------------------------------------------------
Returns the number of frames in clips
file_path: path to videos to count frames
"""
def get_frame_values(file_path):
    frame_rates = []
    for index, filename in enumerate(os.listdir(file_path)):
        # If video file
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            input_movie = init_video(file_path + filename)
            length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rates = frame_rates + [length]

"""
Flip transformation on a frame
-----------------------------------------------------------
img: frame to flip
"""
def flip_img(img):
    horizontal_img = img.copy()
    # flip img horizontally
    horizontal_img = cv2.flip( img, 0 )
    return horizontal_img

"""
Retrive data from folders
-----------------------------------------------------------
folder: folder with video files
rgb: select between RGB and MV files
"""
def retrieve_data(folder, rgb=True, mv_type='mag'):
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
            mvs = []
            with open(folder + filename, 'r') as f:
                reader = csv.reader(f)
                for i in reader:
                    frame = []
                    for row in list(i):
                        row_l = ast.literal_eval(row)
                        frame.append(row_l)
                    mvs.append(np.array(frame, dtype=np.float32))
            mv_arr = np.array(mvs)
            data.append(mv_arr)

    return data

"""
Split videos by defined number of frames
-----------------------------------------------------------
data: video to split
chunk: number of frames per split
"""
def split_frames(data, chunk):
    split_fs = []

    for video in data:
        split_fs = split_fs + [video[i:i + chunk] for i in range(0, len(video), chunk)]

    split_fs = [item for item in split_fs if len(item) == chunk]

    return split_fs
