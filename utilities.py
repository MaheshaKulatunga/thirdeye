import cv2
import os


def init_video(filepath):

    vid = cv2.VideoCapture(filepath)

    return vid


def get_frame_values(file_path):
    frame_rates = []
    for index, filename in enumerate(os.listdir(file_path)):
        # If video file
        if filename.endswith(".mp4"):
            input_movie = init_video(file_path + filename)
            length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rates = frame_rates + [length]
            print(length)
    print(min(frame_rates))


# This is now done within the raw file processing
# def standardise_fps(file_path):
#     for index, filename in enumerate(os.listdir(file_path)):
#         # If video file
#         if filename.endswith(".mp4"):
#             print(file_path + filename)
#             print(TRAIN_FPS_DEEPFAKES + filename)
#             command = "ffmpeg -i {} -r 22 -y {}".format(file_path + filename, TRAIN_FPS_DEEPFAKES + filename)
#             subprocess.call(command, shell=True)
