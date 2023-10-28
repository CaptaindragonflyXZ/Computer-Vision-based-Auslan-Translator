import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math as math
import time


SAVE_PATH = 'YUPENG_test_video\\num_ten'


COU_INIT = 0
CAMERA_HEIGHT = 500
CAMERA_WIDTH  = 650

WHITE_COLOR = (255, 255, 255)   # hand frame
RED_COLOR   = (0, 0, 255)       # joint points

BLUE_COLOR  = (255, 0, 0)       # thumb
GREEN_COLOR = (0, 255, 0)       # index finger
YELLOW_COLOR = (255,255,0)      # middle finger
CYAN_COLOR = (0,255,255)        # ring finger
MEGENTA_COLOR = (255,0,255) 	# little finger

# ================================================================== 
folder_name = SAVE_PATH
sign_name = 'sample'
# ================================================================== 


def main():

    print(f'save path is {SAVE_PATH}')
    key_1 = 's'
    key_2 = 'd'
    my_key = key_1
    
    
    feature_list = []
    cap = cv2.VideoCapture(0)
    detector_1 = HandDetector(maxHands=2)
    detector_2 = HandDetector(maxHands=2)
    cou = COU_INIT

    while True:
        success, img_ori = cap.read()
        img_ori = cv2.flip(img_ori, 1)
        img_ori_copy_0 = np.copy(img_ori)
        img_ori_copy_1 = np.copy(img_ori)
        hands, img = detector_1.findHands(img_ori)

        if len(hands) == 2:
            if hands[0]['type'] == hands[1]['type']:
                if hands[0]['center'][0] < hands[1]['center'][0]:
                    hands[0]['type'] = "Right"
                    hands[1]['type'] = "Left"
                else:
                    hands[0]['type'] = "Left"
                    hands[1]['type'] = "Right"

        feature_array = feature_extraction(hands)

        feature_array_255 = (feature_array * 255).astype(np.uint8)
        feature_array_255_large = cv2.resize(feature_array_255, (525, 100))
        

        cv2.imshow('img_ori', img_ori)
        cv2.imshow('feature_array', feature_array_255_large)


        # save the videossssss
        key = cv2.waitKey(1)
        if key == ord(my_key):
            # print('s is pressed')

            feature_list.append(feature_array_255)

            if len(feature_list) > 45:

                output_file = f'{folder_name}/{sign_name}_{str(cou)}.avi'

                frame_rate = 20
                convert_images_to_avi(feature_list, output_file, frame_rate)
                feature_list = []
                print(f'sample {cou} is recorded')
                
                cou += 1

                if my_key == key_1:
                    my_key = key_2
                elif my_key == key_2:
                    my_key = key_1

                print(f'key is switched to {my_key}')
                

# function used to extract the hand features
def feature_extraction(hands_data):
    feature_array = np.zeros((4, 21), dtype=np.float64)
    if (len(hands_data) != 1) and (len(hands_data) != 2):
        pass

    elif hands_data and len(hands_data) == 1:
        hand = hands_data[0]
        if hand['type'] == 'Left':  
            landmarks = hand['lmList']
            for idx in range(len(landmarks)):
                xx, yy, zz = landmarks[idx]
                feature_array[0,idx] = xx
                feature_array[1,idx] = yy

        elif hand['type'] == 'Right':
            landmarks = hand['lmList']
            for idx in range(len(landmarks)):
                xx, yy, zz = landmarks[idx]
                feature_array[2,idx] = xx
                feature_array[3,idx] = yy
            
    # two hand case (alphabet)
    elif hands_data and len(hands_data) == 2:
        for hand in hands_data:

            if hand['type'] == 'Left':
                landmarks = hand['lmList']
                for idx in range(len(landmarks)):
                    xx, yy, zz = landmarks[idx]
                    feature_array[0,idx] = xx
                    feature_array[1,idx] = yy
            
            if hand['type'] == 'Right':
                landmarks = hand['lmList']
                for idx in range(len(landmarks)):
                    xx, yy, zz = landmarks[idx]
                    feature_array[2,idx] = xx
                    feature_array[3,idx] = yy

    # normalize the data
    feature_array_norm = normalize_array(feature_array)
    return feature_array_norm



# normalize the data
def normalize_array(input_array):

    # input_array has the size of 4 x 21
    output_array = np.zeros((4, 21), dtype=np.float64)

    no_hand = np.all(input_array[0, :] == 0) and np.all(input_array[2, :] == 0)
    is_left_hand = np.all(input_array[2, :] == 0)
    is_right_hand = np.all(input_array[0, :] == 0)

    if no_hand:
        pass
    elif is_left_hand and (not is_right_hand):
        # Find  min values for zero position
        min_x = np.min(input_array[0, :])
        min_y = np.min(input_array[1, :])

        # compute the relative position of from each hand point to the zero point
        input_array[0, :] -= min_x
        input_array[1, :] -= min_y

        # normalization
        row = input_array[0, :]
        output_array[0,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
        row = input_array[1, :]
        output_array[1,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
    
    elif is_right_hand and (not is_left_hand):
        # Find  min values for zero position
        min_x = np.min(input_array[2, :])
        min_y = np.min(input_array[3, :])

        # compute the relative position of from each hand point to the zero point
        input_array[2, :] -= min_x
        input_array[3, :] -= min_y

        # normalization
        row = input_array[2, :]
        output_array[2,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
        row = input_array[3, :]
        output_array[3,:] = (row - np.min(row)) / (np.max(row) - np.min(row))

    else:

        # Find max and min values 
        min_x_1 = np.min(input_array[0, :])
        min_y_1 = np.min(input_array[1, :])
        min_x_2 = np.min(input_array[2, :])
        min_y_2 = np.min(input_array[3, :])

        # compute the zero posiiton 
        min_x = min(min_x_1, min_x_2)
        min_y = min(min_y_1, min_y_2)

        # compute the relative position of from each hand point to the zero point
        input_array[0, :] -= min_x
        input_array[1, :] -= min_y
        input_array[2, :] -= min_x
        input_array[3, :] -= min_y

        # normalization
        row = input_array[0, :]
        output_array[0,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
        row = input_array[1, :]
        output_array[1,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
        row = input_array[2, :]
        output_array[2,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
        row = input_array[3, :]
        output_array[3,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
    return output_array



def convert_images_to_avi(image_list, output_filename, fps=20):

    # Get the shape of the first image
    first_image = image_list[0]
    height, width = first_image.shape


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)

    
    # Write the image to the video file
    for image in image_list:
        # print(image.shape)
        video_writer.write(image)

    # Release (save) the video
    video_writer.release()
    print("Images are converted to Video successfully.")


if __name__ == '__main__':
    main()
    # input_array = np.array([
    # [39, 81, 19, 64, 18, 39, 64, 33, 82, 70,  3, 90, 71, 88, 30, 60, 84, 52,  4, 85, 76],
    # [51, 73, 33, 27, 62, 24, 17,  0,  8, 28, 15, 70, 39, 86, 88, 87, 94, 52, 70, 26, 18],
    # [78, 54, 34, 42, 94, 65, 11,  8, 69, 20, 71, 99, 31, 53,  6, 34,  7,  5, 66, 43, 65],
    # [60, 20, 15, 22, 54, 11, 97, 60, 82, 12, 19, 75, 26, 50, 42, 11, 61, 49, 17, 15, 20]])

    # input_array = np.array([
    # [39, 81, 19, 64, 18, 39, 64, 33, 82, 70,  3, 90, 71, 88, 30, 60, 84, 52,  4, 85, 76],
    # [51, 73, 33, 27, 62, 24, 17,  0,  8, 28, 15, 70, 39, 86, 88, 87, 94, 52, 70, 26, 18],
    # [0 ,  0,  0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0, 0, 0],
    # [0 ,  0,  0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0, 0, 0]])

    # print(input_array)
    # output_array = normalize_array(input_array)
    # print(output_array)