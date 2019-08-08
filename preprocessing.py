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
import csv
import sys

class Preprocessor:

    """ Initialize Class """
    def __init__(self):
        pass

    """ Carry out preprocessing """
    def preprocess(self, split):
        try:
            if split == 1:
                self.handle_train_files(1)
            elif split == 2:
                self.handle_test_files(2)
            else:
                self.handle_unknown_files(3)
        except:
            print("Oops!",sys.exc_info()[0],"occured. Ensure files for preprocessing are valid")

    """ Get the largest face in a clip to ensur entire face is always visible after cropping """
    def get_largest_face_size(self, video):
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

    """ Split raw videos into a given number of frames """
    def split_raw_videos(self, clip_size, file_path, fps_path, output_path, split):
        raw_file_list = os.listdir(file_path)
        # Filter only new Files
        old_files = []
        old_files_path = '{}processed_files_{}.csv'.format(constants.DATA, split)
        exists = os.path.isfile(old_files_path)
        if exists:
            with open(old_files_path, 'r') as f:
                reader = csv.reader(f)
                for i in reader:
                    old_files.append(i[0])
            new_files = [x for x in raw_file_list if x not in old_files]
        else:
            new_files = raw_file_list

        # Loop through files in folder
        for index, filename in enumerate(new_files):
            # If video file
            if filename.endswith(".mp4"):
                video_filetype = "mp4"


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
                    # Store filename in folder
                    with open(old_files_path, 'a') as f:
                        f.write("%s\n" % filename)
                else:
                    print('Warning missing file {}'.format(filename))
            else:
                print('Warning: Incompatible file {}'.format(filename))
        print('File split Complete')

    """ Crop videos given a bounding box """
    def crop_videos(self, file_path, output_folder, box_bias, box_size, frames):
        # Loop through files in folder
        for index, filename in enumerate(os.listdir(file_path)):
            # If video file
            if filename.endswith(".mp4") or filename.endswith(".avi"):
                self.facial_extraction(file_path, filename, output_folder, box_bias, box_size, frames)

    """ Detect and crop faces from clips """
    def facial_extraction(self, folder, file_name, output_folder, box_bias, box_size, frames):
        print('Dealing with video {}'.format(file_name))
        input_movie = utilities.init_video(folder + file_name)

        # ffmpeg -y -r 24 -i seeing_noaudio.mp4 seeing.mp4

        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
        # width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        # height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

        fcc = "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Initialize some variables
        frame_number = 0
        count = 0
        largest_face_width, largest_face_height = self.get_largest_face_size(input_movie)
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
            if len(frame_list) >= (frames * 0.75):
                output_movie = cv2.VideoWriter(output_folder + file_name, fourcc, length, (box_size, box_size))

                print('Duplicating frames for video {}'.format(file_name))
                frame_list = frame_list + [frame_list[0]] * (frames - len(frame_list))
                for f in range(frames):
                    print("Writing frame {} / {}".format(f+1, length))
                    output_movie.write(frame_list[f])
            else:
                print('Discarding invalid video {}'.format(file_name))

        # All done!
        input_movie.release()
        cv2.destroyAllWindows()

    """ Extract motion vectors from clips """
    def motion_vector_extraction(self, input_folder, output_folder, frames, box_size):
        # Loop through files in folder
        for index, filename in enumerate(os.listdir(input_folder)):
            # If video file
            if filename.endswith(".mp4"):
                print('Dealing with video {}'.format(filename))
                input_movie = utilities.init_video(input_folder + filename)

                length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

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

                mag = mag_obj.tolist()
                # ang = ang_obj.tolist()

                # Write the resulting frames to the output video file
                if len(frame_list) == (frames-1):
                    print("Writing vectors for {}".format(filename))
                    with open(output_folder + filename + '.csv', "w+") as outfile:
                        writer = csv.writer(outfile, delimiter=',')
                        for f in mag:
                            writer.writerow(f)
                else:
                    print('Discarding invalid video {}'.format(filename))

                input_movie.release()
                cv2.destroyAllWindows()

    """ Preprocesses training files """
    def handle_train_files(self, split):
        print('Preprocessing training files')
        print("Looking for raw videos")
        if len(os.listdir(constants.RAW_DEEPFAKES)) == 1:
            print('No New Raw Videos Found!')
        else:
            start_time = time.time()
            subprocess.call("chmod +x {}rename.sh".format(constants.RAW_DEEPFAKES), shell=True)
            subprocess.call("sh {}rename.sh {}".format(constants.RAW_DEEPFAKES, constants.RAW_DEEPFAKES), shell=True)
            self.split_raw_videos(1, constants.RAW_DEEPFAKES, constants.TRAIN_FPS_DEEPFAKES , constants.TRAIN_DEEPFAKES, split)
            utilities.clear_folder(constants.TRAIN_FPS_DEEPFAKES)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking for videos to crop')
        if len(os.listdir(constants.TRAIN_DEEPFAKES)) == 2:
            print('Can\'t find videos to crop!')
        else:
            utilities.get_frame_values(constants.TRAIN_DEEPFAKES)
            utilities.get_frame_values(constants.TRAIN_FPS_DEEPFAKES)

            start_time = time.time()
            self.crop_videos(constants.TRAIN_DEEPFAKES, constants.TRAIN_SEPARATED_DF_FACES, 20, 100, 20)
            utilities.clear_folder(constants.TRAIN_DEEPFAKES)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking to extract motion vectors')
        if len(os.listdir(constants.TRAIN_SEPARATED_DF_FACES)) == 0:
            print('Can\'t find videos to extract motion vectors from!')
        else:
            start_time = time.time()
            self.motion_vector_extraction(constants.TRAIN_SEPARATED_DF_FACES, constants.TRAIN_MV_DF_FACES, 20, 50)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print("Looking for raw videos")
        if len(os.listdir(constants.RAW_REAL)) == 1:
            print('No Raw Videos Found!')
        else:
            start_time = time.time()
            subprocess.call("chmod +x {}rename.sh".format(constants.RAW_REAL), shell=True)
            subprocess.call("sh {}rename.sh {}".format(constants.RAW_REAL, constants.RAW_REAL), shell=True)
            self.split_raw_videos(1, constants.RAW_REAL, constants.TRAIN_FPS_REAL , constants.TRAIN_REAL, split)
            utilities.clear_folder(constants.TRAIN_FPS_REAL)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking for videos to crop')
        if len(os.listdir(constants.TRAIN_REAL)) == 2:
            print('Can\'t find videos to crop!')
        else:
            utilities.get_frame_values(constants.TRAIN_REAL)
            utilities.get_frame_values(constants.TRAIN_FPS_REAL)

            start_time = time.time()
            self.crop_videos(constants.TRAIN_REAL, constants.TRAIN_SEPARATED_REAL_FACES, 20, 100, 20)
            utilities.clear_folder(constants.TRAIN_REAL)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking to extract motion vectors')
        if len(os.listdir(constants.TRAIN_SEPARATED_REAL_FACES)) == 0:
            print('Can\'t find videos to extract motion vectors from!')
        else:
            start_time = time.time()
            self.motion_vector_extraction(constants.TRAIN_SEPARATED_REAL_FACES, constants.TRAIN_MV_REAL_FACES, 20, 50)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

    """ Preprocesses testing files """
    def handle_test_files(self, split):
        print('Preprocessing videos for testing')
        print("Looking for raw videos")
        if len(os.listdir(constants.TEST_RAW_DEEPFAKES)) == 1:
            print('No New Raw Videos Found!')
        else:
            start_time = time.time()
            subprocess.call("chmod +x {}rename.sh".format(constants.TEST_RAW_DEEPFAKES), shell=True)
            subprocess.call("sh {}rename.sh {}".format(constants.TEST_RAW_DEEPFAKES, constants.TEST_RAW_DEEPFAKES), shell=True)
            self.split_raw_videos(1, constants.TEST_RAW_DEEPFAKES, constants.TEST_FPS_DEEPFAKES , constants.TEST_DEEPFAKES, split)
            utilities.clear_folder(constants.TEST_FPS_DEEPFAKES)

            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking for videos to crop')
        if len(os.listdir(constants.TEST_DEEPFAKES)) == 2:
            print('Can\'t find videos to crop!')
        else:
            utilities.get_frame_values(constants.TEST_DEEPFAKES)
            utilities.get_frame_values(constants.TEST_FPS_DEEPFAKES)

            start_time = time.time()
            self.crop_videos(constants.TEST_DEEPFAKES, constants.TEST_SEPARATED_DF_FACES, 20, 100, 20)
            utilities.clear_folder(constants.TEST_DEEPFAKES)

            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking to extract motion vectors')
        if len(os.listdir(constants.TEST_SEPARATED_DF_FACES)) == 0:
            print('Can\'t find videos to extract motion vectors from!')
        else:
            start_time = time.time()
            self.motion_vector_extraction(constants.TEST_SEPARATED_DF_FACES, constants.TEST_MV_DF_FACES, 20, 50)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print("Looking for raw videos")
        if len(os.listdir(constants.TEST_RAW_REAL)) == 1:
            print('No Raw Videos Found!')
        else:
            start_time = time.time()
            subprocess.call("chmod +x {}rename.sh".format(constants.TEST_RAW_REAL), shell=True)
            subprocess.call("sh {}rename.sh {}".format(constants.TEST_RAW_REAL, constants.TEST_RAW_REAL), shell=True)
            self.split_raw_videos(1, constants.TEST_RAW_REAL, constants.TEST_FPS_REAL , constants.TEST_REAL, split)
            utilities.clear_folder(constants.TEST_FPS_REAL)

            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking for videos to crop')
        if len(os.listdir(constants.TEST_REAL)) == 2:
            print('Can\'t find videos to crop!')
        else:
            utilities.get_frame_values(constants.TEST_REAL)
            utilities.get_frame_values(constants.TEST_FPS_REAL)

            start_time = time.time()
            self.crop_videos(constants.TEST_REAL, constants.TEST_SEPARATED_REAL_FACES, 20, 100, 20)
            utilities.clear_folder(constants.TEST_REAL)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking to extract motion vectors')
        if len(os.listdir(constants.TEST_SEPARATED_REAL_FACES)) == 0:
            print('Can\'t find videos to extract motion vectors from!')
        else:
            start_time = time.time()
            self.motion_vector_extraction(constants.TEST_SEPARATED_REAL_FACES, constants.TEST_MV_REAL_FACES, 20, 50)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

    """ Preprocesses unknown files """
    def handle_unknown_files(self, split):
        print('Preprocessing files to classify')
        print('Looking for raw videos')
        if len(os.listdir(constants.UNKNOWN_RAW)) == 0:
            print('No Raw Videos Found!')
        else:
            start_time = time.time()
            subprocess.call("chmod +x {}rename.sh".format(constants.UNKNOWN_RAW), shell=True)
            subprocess.call("sh {}rename.sh {}".format(constants.UNKNOWN_RAW, constants.UNKNOWN_RAW), shell=True)
            self.split_raw_videos(1, constants.UNKNOWN_RAW, constants.UNKNOWN_FPS , constants.UNKNOWN_CLIPS, split)
            print('Clearing folders')
            utilities.clear_folder(constants.UNKNOWN_FPS)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))

        print('Looking for videos to crop')
        if len(os.listdir(constants.UNKNOWN_CLIPS)) == 2:
            print('Can\'t find videos to crop!')
        else:
            utilities.get_frame_values(constants.UNKNOWN_CLIPS)
            utilities.get_frame_values(constants.UNKNOWN_FPS)

            start_time = time.time()
            self.crop_videos(constants.UNKNOWN_CLIPS, constants.UNKNOWN_SEP, 20, 100, 20)
            utilities.clear_folder(constants.UNKNOWN_CLIPS)
            time_taken = round(((time.time() - start_time) / 60.0), 2)
            print("--- Completed in {} minutes ---".format(time_taken))
