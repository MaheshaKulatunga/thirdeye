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


def facial_extraction(file_path):
    input_movie = cv2.VideoCapture(file_path+'00.mp4')
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(length)
    width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(TRAIN_SEPERATED_DF_FACES + 'output.mp4', fourcc, length, (155, 155))

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    count = 0
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

        top, right, bottom, left = face_locations[0]

        if (right - left) < 155:
            right = right + (155 - (right - left))

        if (bottom - top) < 155:
            bottom = bottom + (155 - (bottom - top))

        crop_img = frame[top:bottom, left:right]
        frame = frame[top:bottom, left:right]

        print(crop_img.shape)
        print(frame.shape)
        print(crop_img.size)
        print(frame.size)

        cv2.imwrite(TRAIN_SEPERATED_DF_FACES + str(count) + "test.png", crop_img)

        # Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(frame)

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

    facial_extraction(TRAIN_DEEPFAKES)
