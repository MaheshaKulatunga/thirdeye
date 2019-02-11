import time
import numpy as np
# module load cs909-python
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

# Constants
# File Paths
RAW_DEEPFAKES = './Data/Raw/DF/'
RAW_REAL = './Data/Raw/REAL/'

TRAIN_DEEPFAKES = './Data/Train/DF/'
TRAIN_REAL = './Data/Train/REAL/'

TEST_DEEPFAKES = './Data/Test/DF/'
TEST_REAL = './Data/Test/Real/'


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
                    output_video_path = '{}{}{}.mp4'.format(TRAIN_DEEPFAKES, index,clip)
                    start = clip
                    end = clip+1
                    new = video.subclip(start, end)
                    new.write_videofile(output_video_path, audio_codec='aac')
        else:
            print('Warning: Incompatible file')
    print('File split Complete')

def facial_extraction(file_path):
    pass

if __name__ == "__main__":
    split_raw_videos(1, RAW_DEEPFAKES)
