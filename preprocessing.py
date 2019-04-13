import time
import numpy as np
# module load cs909-python
import os
import subprocess
import face_recognition
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
import utilities
import constants
import json


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
        if face_locations_init:
            top_i, right_i, bottom_i, left_i = face_locations_init[0]

            height_i = bottom_i-top_i
            width_i = right_i-left_i
            if height_i > largest_face_height_i:
                largest_face_height_i = height_i
            if width_i > largest_face_width_i:
                largest_face_width_i = width_i

        count += 1
    return largest_face_width_i, largest_face_height_i


def split_raw_videos(clip_size, file_path, fps_path, output_path):
    # Loop through files in folder
    for index, filename in enumerate(os.listdir(file_path)):
        # If video file
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            if filename.endswith(".mp4"):
                video_filetype = "mp4"
            if filename.endswith(".avi"):
                # video_filetype = "avi"
                video_filetype = "avi"


            command = "ffmpeg -i {} -r 20 -y {}".format(file_path + filename, fps_path + filename)
            subprocess.call(command, shell=True)

            input_video_path = fps_path + filename
            # Import video
            if os.path.isfile(input_video_path):
                with VideoFileClip(input_video_path) as video:
                    # Get video duration and calculate number of possible clips
                    clip_count = int(video.duration/clip_size)
                    # Split each clip and save
                    for clip in range(clip_count):
                        output_video_path = '{}{}{}.{}'.format(output_path, index, clip, video_filetype) #TODO CONVERT HERE
                        start = clip
                        end = clip + clip_size
                        new = video.subclip(start, end)
                        new.write_videofile(output_video_path, audio=False, codec='libx264')
            else:
                print('Warning missing file {}'.format(filename))
        else:
            print('Warning: Incompatible file')
    print('File split Complete')


def crop_videos(file_path, output_folder, box_bias, box_size, frames):
    # Loop through files in folder
    for index, filename in enumerate(os.listdir(file_path)):
        # If video file
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            facial_extraction(file_path, filename, output_folder, box_bias, box_size, frames)


def facial_extraction(folder, file_name, output_folder, box_bias, box_size, frames):
    print('Dealing with video {}'.format(file_name))
    input_movie = utilities.init_video(folder + file_name)

    # ffmpeg -y -r 24 -i seeing_noaudio.mp4 seeing.mp4

    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    # width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    # height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    if file_name.endswith(".mp4"):
        fcc = "mp4v"
    if file_name.endswith(".avi"):
        fcc = "mp4v"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Initialize some variables
    frame_number = 0
    count = 0
    largest_face_width, largest_face_height = get_largest_face_size(input_movie)
    frame_list = []

    input_movie = utilities.init_video(folder + file_name)

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
        # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_locations:
            print('Found face in frame {} of video {}.'.format(frame_number, file_name))
            top, right, bottom, left = face_locations[0]

            if (right - left) < largest_face_width:
                right = right + (largest_face_width - (right - left))

            if (bottom - top) < largest_face_height:
                bottom = bottom + (largest_face_height - (bottom - top))

            frame = frame[top - box_bias:bottom + box_bias, left - box_bias:right + box_bias]

            try:
                frame = cv2.resize(frame, (box_size, box_size), interpolation=cv2.INTER_LINEAR)
                frame_list = frame_list + [frame]

            except Exception as e:
                print(str(e))

        else:
            print('Warning: Frame {} with missing face in video {}'.format(frame_number, file_name))

        # Frames as images?
        # crop_img = frame[top:bottom, left:right]
        # cv2.imwrite(TRAIN_SEPARATED_DF_FACES + str(count) + "test.png", crop_img)
        count += 1

    # Write the resulting frames to the output video file
    if len(frame_list) == frames:
        output_movie = cv2.VideoWriter(output_folder + file_name, fourcc, length, (box_size, box_size))

        for f in range(frames):
            print("Writing frame {} / {}".format(f+1, length))
            output_movie.write(frame_list[f])
    else:
        print('Discarding invalid video {}'.format(file_name))

    # All done!
    input_movie.release()
    cv2.destroyAllWindows()


def motion_vector_extraction(input_folder, output_folder, frames, box_size):
    # Loop through files in folder
    for index, filename in enumerate(os.listdir(input_folder)):
        # If video file
        if (filename.endswith(".mp4") or filename.endswith(".avi")):
            print('Dealing with video {}'.format(filename))
            input_movie = utilities.init_video(input_folder + filename)

            length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

            if filename.endswith(".mp4"):
                fcc = "mp4v"
            if filename.endswith(".avi"):
                fcc = "mp4v"

            fourcc = cv2.VideoWriter_fourcc(*fcc)

            frame_list = []
            ang_obj = []
            mag_obj = []

            ret, frame1 = input_movie.read()
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255
            while (1):
                ret, frame2 = input_movie.read()

                if not ret:
                    break

                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # cv2.imshow('frame2', rgb)
                # k = cv2.waitKey(30) & 0xff
                # print(k)
                # if k == 27:
                #     break
                # elif k == ord('s'):
                # cv2.imwrite('opticalfb.png', frame2)
                frame_list = frame_list + [rgb]
                ang_obj = ang_obj + [ang]
                mag_obj = mag_obj + [mag]
                # cv2.imwrite('test.png', rgb)
                prvs = next

            ang_obj = np.array(ang_obj)
            mag_obj = np.array(mag_obj)

            motion_obj = {'mag': mag_obj.tolist(), 'ang': ang_obj.tolist()}

            # Write the resulting frames to the output video file
            if len(frame_list) == (frames-1):
                # output_movie = cv2.VideoWriter(output_folder + filename, fourcc, length, (100, 100))
                # for f in range(frames-1):
                #     print("Writing frame {} / {}".format(f + 1, length))
                #     output_movie.write(frame_list[f])

                print("Writing vectors for {}".format(filename))
                with open(output_folder + filename + '.json', 'w') as outfile:
                    json.dump(motion_obj, outfile)


            else:
                print('Discarding invalid video {}'.format(filename))

            input_movie.release()
            cv2.destroyAllWindows()


def handle_files():
    if len(os.listdir(constants.TRAIN_REAL)) <= 2:
        print("Looking for raw videos")
        if len(os.listdir(constants.RAW_REAL)) == 0:
            print('No Raw Videos Found!')
        else:
            start_time = time.time()
            split_raw_videos(1, constants.RAW_REAL, constants.TRAIN_FPS_REAL , constants.TRAIN_REAL)
            print("--- %s seconds ---" % (time.time() - start_time))

    else:
        print('Training Videos Detected')

    if len(os.listdir(constants.TRAIN_SEPARATED_REAL_FACES)) == 0:
        print('Looking for videos to crop')
        if len(os.listdir(constants.TRAIN_REAL)) == 1:
            print('Can\'nt find videos to crop!')
        else:
            utilities.get_frame_values(constants.TRAIN_REAL)
            utilities.get_frame_values(constants.TRAIN_FPS_REAL)

            start_time = time.time()
            crop_videos(constants.TRAIN_REAL, constants.TRAIN_SEPARATED_REAL_FACES, 20, 100, 20)
            print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print('Cropped videos detected')
    if len(os.listdir(constants.TRAIN_MV_REAL_FACES)) == 0:
        print('Looking to extract motion vectors')
        if len(os.listdir(constants.TRAIN_SEPARATED_REAL_FACES)) == 1:
            print('Can\'nt find videos to extract motion vectors from!')
        else:
            start_time = time.time()
            motion_vector_extraction(constants.TRAIN_SEPARATED_REAL_FACES, constants.TRAIN_MV_REAL_FACES, 20, 50)
            print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print('Motion vectors detected')
        motion_vector_extraction(constants.TRAIN_SEPARATED_REAL_FACES, constants.TRAIN_MV_REAL_FACES, 20, 50)

def handle_test_files():
    if len(os.listdir(constants.TEST_DEEPFAKES)) <= 3:
        print("Looking for raw videos")
        if len(os.listdir(constants.RAW_TEST)) == 0:
            print('No Raw Videos Found!')
        else:
            start_time = time.time()
            split_raw_videos(1, constants.RAW_TEST, constants.TEST_FPS_DEEPFAKES , constants.TEST_DEEPFAKES)
            print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print('Training Videos Detected')

    if len(os.listdir(constants.TEST_SEPARATED_DF_FACES)) == 0:
        print('Looking for videos to crop')
        if len(os.listdir(constants.TEST_DEEPFAKES)) == 1:
            print('Can\'nt find videos to crop!')
        else:
            utilities.get_frame_values(constants.TEST_DEEPFAKES)
            utilities.get_frame_values(constants.TEST_FPS_DEEPFAKES)

            start_time = time.time()
            crop_videos(constants.TEST_DEEPFAKES, constants.TEST_SEPARATED_DF_FACES, 20, 100, 20)
            print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print('Cropped videos detected')

    if len(os.listdir(constants.TEST_MV_DF_FACES)) == 0:
        print('Looking to extract motion vectors')
        if len(os.listdir(constants.TEST_SEPARATED_DF_FACES)) == 1:
            print('Can\'nt find videos to extract motion vectors from!')
        else:
            start_time = time.time()
            motion_vector_extraction(constants.TEST_SEPARATED_DF_FACES, constants.TEST_MV_DF_FACES, 20, 50)
            print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print('Motion vectors detected')
        motion_vector_extraction(constants.TEST_SEPARATED_DF_FACES, constants.TEST_MV_DF_FACES, 20, 50)

if __name__ == "__main__":
    # if len(os.listdir(constants.TRAIN_DEEPFAKES)) <= 1:
    #     print("Looking for raw videos")
    #     if len(os.listdir(constants.RAW_DEEPFAKES)) == 0:
    #         print('No Raw Videos Found!')
    #     else:
    #         start_time = time.time()
    #         split_raw_videos(1, constants.RAW_DEEPFAKES, constants.TRAIN_FPS_DEEPFAKES , constants.TRAIN_DEEPFAKES)
    #         print("--- %s seconds ---" % (time.time() - start_time))
    #
    # else:
    #     print('Training Videos Detected')
    #
    # if len(os.listdir(constants.TRAIN_SEPARATED_DF_FACES)) == 0:
    #     print('Looking for videos to crop')
    #     if len(os.listdir(constants.TRAIN_DEEPFAKES)) == 1:
    #         print('Can\'nt find videos to crop!')
    #     else:
    #         utilities.get_frame_values(constants.TRAIN_DEEPFAKES)
    #         utilities.get_frame_values(constants.TRAIN_FPS_DEEPFAKES)
    #
    #         start_time = time.time()
    #         crop_videos(constants.TRAIN_DEEPFAKES, constants.TRAIN_SEPARATED_DF_FACES, 20, 100, 20)
    #         print("--- %s seconds ---" % (time.time() - start_time))
    # else:
    #     print('Cropped videos detected')
    #
    # if len(os.listdir(constants.TRAIN_MV_DF_FACES)) == 0:
    #     print('Looking to extract motion vectors')
    #     if len(os.listdir(constants.TRAIN_SEPARATED_DF_FACES)) == 1:
    #         print('Can\'nt find videos to extract motion vectors from!')
    #     else:
    #         start_time = time.time()
    #         motion_vector_extraction(constants.TRAIN_SEPARATED_DF_FACES, constants.TRAIN_MV_DF_FACES, 20, 50)
    #         print("--- %s seconds ---" % (time.time() - start_time))
    # else:
    #     print('Motion vectors detected')
    #     motion_vector_extraction(constants.TRAIN_SEPARATED_DF_FACES, constants.TRAIN_MV_DF_FACES, 20, 50)
    #
    handle_files()
    # handle_test_files()
