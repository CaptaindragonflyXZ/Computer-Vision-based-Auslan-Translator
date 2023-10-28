import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math as math
import time

SAVE_PATH = 'Data_ori\\Auslan_Dataset\\number_ten'
SAVE_PATH = 'YUPENG_test_video\\txt_how_are_you'
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


def write_hand_points(hand_point_img, lm_list):
    pass


def write_hand_lines(hand_point_img, lm_list):

    # lines form a hand shape
    cv2.line(hand_point_img, lm_list[0],  lm_list[1],  RED_COLOR, 5)
    cv2.line(hand_point_img, lm_list[0],  lm_list[5],  RED_COLOR, 5)
    cv2.line(hand_point_img, lm_list[0],  lm_list[9],  RED_COLOR, 5)
    cv2.line(hand_point_img, lm_list[0],  lm_list[13], RED_COLOR, 5)
    cv2.line(hand_point_img, lm_list[0],  lm_list[17], RED_COLOR, 5)
    cv2.line(hand_point_img, lm_list[5],  lm_list[9],  RED_COLOR, 5)
    cv2.line(hand_point_img, lm_list[9],  lm_list[13], RED_COLOR, 5)
    cv2.line(hand_point_img, lm_list[13], lm_list[17], RED_COLOR, 5)

    # one hand is 12 pixel
    # thumb
    cv2.line(hand_point_img, lm_list[1], lm_list[2], WHITE_COLOR, 8)
    cv2.line(hand_point_img, lm_list[2], lm_list[3], WHITE_COLOR, 8)
    cv2.line(hand_point_img, lm_list[3], lm_list[4], WHITE_COLOR, 8)

    # index finger
    cv2.line(hand_point_img, lm_list[5], lm_list[6], GREEN_COLOR, 8)
    cv2.line(hand_point_img, lm_list[6], lm_list[7], GREEN_COLOR, 8)
    cv2.line(hand_point_img, lm_list[7], lm_list[8], GREEN_COLOR, 8)

    # middle finger
    cv2.line(hand_point_img, lm_list[9],  lm_list[10], YELLOW_COLOR, 8)
    cv2.line(hand_point_img, lm_list[10], lm_list[11], YELLOW_COLOR, 8)
    cv2.line(hand_point_img, lm_list[11], lm_list[12], YELLOW_COLOR, 8)

    # ring finger
    cv2.line(hand_point_img, lm_list[13], lm_list[14], CYAN_COLOR, 8)
    cv2.line(hand_point_img, lm_list[14], lm_list[15], CYAN_COLOR, 8)
    cv2.line(hand_point_img, lm_list[15], lm_list[16], CYAN_COLOR, 8)

    # little finger
    cv2.line(hand_point_img, lm_list[17], lm_list[18], MEGENTA_COLOR, 8)
    cv2.line(hand_point_img, lm_list[18], lm_list[19], MEGENTA_COLOR, 8)
    cv2.line(hand_point_img, lm_list[19], lm_list[20], MEGENTA_COLOR, 8)



def main():
    
    key_1 = 's'
    key_2 = 'd'
    my_key = key_1
    

    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=3)
    cou = COU_INIT

    hand_point_img_list = []


    # ================================================================== 
    folder_name = SAVE_PATH
    sign_name = 'sample'
    # ================================================================== 


    while True:
        success, img_ori = cap.read()
        img_ori = cv2.flip(img_ori, 1)

        hands, img = detector.findHands(img_ori)

        hand_point_img =  np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), np.uint8)
        hand_point_img_cropped = np.zeros((300, 300, 3), np.uint8)

        # one hand case (simple number)
        if hands and len(hands) == 1:
            hand = hands[0]

            # only record hand landmarks
            lm_list = []
            landmarks = hand['lmList']
            for this_lm in landmarks:
                xx, yy, zz = this_lm
                point_2D = (xx, yy)
                lm_list.append(point_2D)
            
            write_hand_lines(hand_point_img, lm_list)
            write_hand_points(hand_point_img, lm_list)
            hand_point_img_cropped = crop_hand_img(hands, hand_point_img)



        # two hand case (alphabet)
        elif hands and len(hands) == 2:

            # only record hand landmarks
            for this_hand in hands:
                lm_list = []
                landmarks = this_hand['lmList']
                for this_lm in landmarks:
                    xx, yy, zz = this_lm
                    point_2D = (xx, yy)
                    lm_list.append(point_2D)
            
                write_hand_lines(hand_point_img, lm_list)
                write_hand_points(hand_point_img, lm_list)    
                hand_point_img_cropped = crop_hand_img(hands, hand_point_img)
            
            hand_point_img_cropped = cv2.resize(hand_point_img_cropped, (300, 300), interpolation = cv2.INTER_AREA)


        # show the images
        cv2.imshow('Image', img)
        cv2.imshow('hand_point_img_cropped', hand_point_img_cropped)


        # save the video
        key = cv2.waitKey(1)
        if key == ord(my_key):

            # print(type(hand_point_img))
            hand_point_img_list.append(hand_point_img_cropped)

            if len(hand_point_img_list) > 60:

                output_file = f'{folder_name}/{sign_name}_{str(cou)}.avi'

                frame_rate = 20
                convert_images_to_avi(hand_point_img_list, output_file, frame_rate)
                hand_point_img_list = []
                print(f'sample {cou} is recorded')
                cou += 1
                if my_key == key_1:
                    my_key = key_2
                elif my_key == key_2:
                    my_key = key_1
                print(f'key is switched to {my_key}')



def convert_images_to_avi(image_list, output_filename, fps=20):

    # Get the shape of the first image
    first_image = image_list[0]
    height, width,_ = first_image.shape

    # try all codec, if works then create VideoWriter object
    for codec in ['XVID', 'MJPG', 'MP4V', 'H264']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if video_writer.isOpened():
            print('codec is', codec)
            break
    
    # Write the image to the video file
    for image in image_list:
        video_writer.write(image)

    # Release (save) the video
    video_writer.release()
    print("Images are converted to Video successfully.")



def crop_hand_img(hands, hand_point_img):

    imgSize = 300
    offset = 20

    # one hand case (number 0 to number 9)
    if len(hands) == 1:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWrite = np.ones((imgSize, imgSize, 3), np.uint8)*0

        if y - offset > 0:
            new_y_small = y - offset
        else: 
            new_y_small = 0
        
        if x - offset > 0:
            new_x_small = x - offset
        else: 
            new_x_small = 0

        imgCrop = hand_point_img[new_y_small: y + offset + h, new_x_small:x + offset + w]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        try:
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWrite[:, wGap:wCal+wGap] = imgResize

            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWrite[hGap:hCal+hGap, :] = imgResize
        except: 
            print('something is wrong')

    #  two hands case (letter a to letter z)
    elif len(hands) == 2:
        hand_1 = hands[0]
        hand_2 = hands[1]
        x1, y1, w1, h1 = hand_1['bbox']
        x2, y2, w2, h2 = hand_2['bbox']

        if x1 <= x2:
            x_min = x1
        else:
            x_min = x2

        if x1+w1 <= x2+w2:
            x_max = x2
            new_x_large = x_max + w2
        else:
            x_max = x1
            new_x_large = x_max + w1
        
        if y1 <= y2:
            y_min = y1

        else:
            y_min = y2

        if y1+h1 <= y2+h2:
            y_max = y2
            new_y_large = y_max + h2
        else:
            y_max = y1
            new_y_large = y_max + h1
        
        imgWrite = np.ones((imgSize, imgSize, 3), np.uint8)*0

        if y_min - offset > 0:
            new_y_small = y_min - offset
        else: 
            new_y_small = 0
        
        if x_min - offset > 0:
            new_x_small = x_min - offset
        else: 
            new_x_small = 0

        imgWrite = hand_point_img[new_y_small: new_y_large + offset, new_x_small:new_x_large + offset]
    
    # no cases that more than two hands
    else:
        print ('cannot more then two hands')
    
    return imgWrite



if __name__ == '__main__':
    main()
