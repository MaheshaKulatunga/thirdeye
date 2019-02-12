import time
import numpy as np
# module load cs909-python
import os
import face_recognition
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

# Constants
# File Paths
RAW_DEEPFAKES = './Data/Raw/DF/'
RAW_REAL = './Data/Raw/REAL/'

TRAIN_DEEPFAKES = './Data/Train/DF/'
TRAIN_REAL = './Data/Train/REAL/'

TEST_DEEPFAKES = './Data/Test/DF/'
TEST_REAL = './Data/Test/Real/'

TRAIN_SEPERATED_DF_FACES = './Data/Train/DF/SEP/'
TRAIN_SEPERATED_REAL_FACES = './Data/Train/REAL/SEP/'

TEST_SEPERATED_DF_FACES = './Data/Test/DF/SEP/'
TEST_SEPERATED_REAL_FACES = './Data/Test/REAL/SEP/'


def split_raw_videos(clip_size, file_path):
    # Loop through files in folder
    for index, filename in enumerate(os.listdir(file_path)):
        # If video file
        if filename.endswith(".mp4"):
            input_video_path = RAW_DEEPFAKES+filename
            # Import video
            with VideoFileClip(input_video_path) as video:
                # Get video duration and calculate number of possible clips
                clip_count = int(video.duration/clip_size)
                # Split each clip and save
                for clip in range(clip_count):
                    output_video_path = '{}{}{}.mp4'.format(TRAIN_DEEPFAKES, index, clip)
                    start = clip
                    end = clip + clip_size
                    new = video.subclip(start, end)
                    new.write_videofile(output_video_path, audio_codec='aac')
        else:
            print('Warning: Incompatible file')
    print('File split Complete')


def get_largest_face_size(video):
    largest_face_height_i = 0
    largest_face_width_i = 0

    count = 0
    while True:
        # Grab a single frame of video
        ret_init, frame_init = video.read()

        # Quit when the input video file ends
        if not ret_init:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame_init = frame_init[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations_init = face_recognition.face_locations(rgb_frame_init)

        top_i, right_i, bottom_i, left_i = face_locations_init[0]

        height_i = bottom_i-top_i
        width_i = right_i-left_i
        if height_i > largest_face_height_i:
            largest_face_height_i = height_i
        if width_i > largest_face_width_i:
            largest_face_width_i = width_i

        count += 1
    return largest_face_width_i, largest_face_height_i


def init_video(filepath):

    vid = cv2.VideoCapture(filepath)

    return vid


def crop_videos(file_path, box_bias):
    # Loop through files in folder
    for index, filename in enumerate(os.listdir(file_path)):
        # If video file
        if filename.endswith(".mp4"):
            facial_extraction(filename, box_bias)


def facial_extraction(file_path, box_bias):
    print('Dealing with video {}'.format(file_path))
    input_movie = init_video(TRAIN_DEEPFAKES + file_path)

    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    # height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize some variables
    frame_number = 0
    count = 0
    largest_face_width, largest_face_height = get_largest_face_size(input_movie)

    input_movie = init_video(TRAIN_DEEPFAKES + file_path)
    output_movie = cv2.VideoWriter(TRAIN_SEPERATED_DF_FACES + file_path, fourcc, length, (largest_face_width, largest_face_height))

    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_locations:
            top, right, bottom, left = face_locations[0]

            if (right - left) < largest_face_width:
                right = right + (largest_face_width - (right - left))

            if (bottom - top) < largest_face_height:
                bottom = bottom + (largest_face_height - (bottom - top))

            frame = frame[top:bottom, left:right]
        else:
            print('Warning: Frame {} with missing face in video {}'.format(frame_number, file_path))

        # Frames as images?
        # crop_img = frame[top:bottom, left:right]
        # cv2.imwrite(TRAIN_SEPERATED_DF_FACES + str(count) + "test.png", crop_img)

        # Write the resulting frames to the output video file
        print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(frame)
        # frame_list = frame_list + [frame]

        count += 1

    # All done!
    input_movie.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(os.listdir(TRAIN_DEEPFAKES)) == 0:
        print("Looking for raw videos")
        if len(os.listdir(RAW_DEEPFAKES)) == 0:
            print('No Raw Videos Found!')
        else:
            split_raw_videos(1, RAW_DEEPFAKES)
    else:
        print('Training Videos Detected')

        crop_videos(TRAIN_DEEPFAKES, 50)
