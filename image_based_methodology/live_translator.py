import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math as math
import time
import torch
import numpy as np
from network import C3D_model
import cv2
import tkinter as tk
from PIL import Image, ImageTk
torch.backends.cudnn.benchmark = True

GUI_GEN = 1
FRAME_NUM = 16
NUM_CLASS = 44
LABEL_FILE = 'Auslan_labels_44.txt'
MODEL_FILE  = 'run\\run_5\\models\\C3D-Auslan_epoch-59.pth.tar'


CAMERA_HEIGHT = 500
CAMERA_WIDTH  = 650

WHITE_COLOR     = (255, 255, 255)   # hand frame
RED_COLOR       = (0, 0, 255)       # joint points

BLUE_COLOR      = (255, 0, 0)       # thumb
GREEN_COLOR     = (0, 255, 0)       # index finger
YELLOW_COLOR    = (255,255,0)      # middle finger
CYAN_COLOR      = (0,255,255)        # ring finger
MEGENTA_COLOR   = (255,0,255)     # little finger

label_cou = 0

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

    # thumb
    cv2.line(hand_point_img, lm_list[1], lm_list[2], WHITE_COLOR, 12)
    cv2.line(hand_point_img, lm_list[2], lm_list[3], WHITE_COLOR, 12)
    cv2.line(hand_point_img, lm_list[3], lm_list[4], WHITE_COLOR, 12)

    # index finger
    cv2.line(hand_point_img, lm_list[5], lm_list[6], GREEN_COLOR, 12)
    cv2.line(hand_point_img, lm_list[6], lm_list[7], GREEN_COLOR, 12)
    cv2.line(hand_point_img, lm_list[7], lm_list[8], GREEN_COLOR, 12)

    # middle finger
    cv2.line(hand_point_img, lm_list[9],  lm_list[10], YELLOW_COLOR, 12)
    cv2.line(hand_point_img, lm_list[10], lm_list[11], YELLOW_COLOR, 12)
    cv2.line(hand_point_img, lm_list[11], lm_list[12], YELLOW_COLOR, 12)

    # ring finger
    cv2.line(hand_point_img, lm_list[13], lm_list[14], CYAN_COLOR, 12)
    cv2.line(hand_point_img, lm_list[14], lm_list[15], CYAN_COLOR, 12)
    cv2.line(hand_point_img, lm_list[15], lm_list[16], CYAN_COLOR, 12)

    # little finger
    cv2.line(hand_point_img, lm_list[17], lm_list[18], MEGENTA_COLOR, 12)
    cv2.line(hand_point_img, lm_list[18], lm_list[19], MEGENTA_COLOR, 12)
    cv2.line(hand_point_img, lm_list[19], lm_list[20], MEGENTA_COLOR, 12)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def convert_images_to_avi(image_list, output_filename, fps=20):

    # Get the shape of the first image
    first_image = image_list[0]
    height, width,_ = first_image.shape

    # try all codec, if works then create VideoWriter object
    for codec in ['XVID', 'MJPG', 'MP4V', 'H264']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if video_writer.isOpened():
            break
    
    # Write the image to the video file
    for image in image_list:
        video_writer.write(image)

    # Release (save) the video
    video_writer.release()


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

        
        # process the limits on x axis
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


        # process the limits on y axis 
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


class GUI():

    # class initialization
    def __init__(self):

        # create a window for GUI display 
        self.window = tk.Tk()
        self.window.title("Auslan Real-time Translator")

        # sentence to display
        self.sentence = tk.StringVar()
        self.sentence.set("Welcome to use the real-time Auslan translator!")

        # set GUI title
        self.label = tk.Label(self.window, text="Auslan Real-time Translator", font=("Helvetica", 30))
        self.label.grid(row=0, columnspan=2)

        self.label = tk.Label(self.window, text="live video stream", font=("Helvetica", 15))
        self.label.grid(row=1, column=0)
        self.label = tk.Label(self.window, text="simplified hand model", font=("Helvetica", 15))
        self.label.grid(row=1, column=1)

        
        # display the video from webcam on the left
        self.left_frame = tk.Canvas(self.window, width=640, height=480)
        self.left_frame.grid(row=2, column=0)
        
        # display the cropped hand image on the right
        self.right_frame = tk.Canvas(self.window, width=360, height=480)
        self.right_frame.grid(row=2, column=1)
        
        # create a label for displaying text
        self.label = tk.Label(self.window, text=" ", font=("Helvetica", 10))
        self.label.grid(row=3, columnspan=2)

        self.label = tk.Label(self.window, textvariable=self.sentence, font=("Helvetica", 33), fg="red")
        self.label.grid(row=4, columnspan=2)

        self.label = tk.Label(self.window, text=" ", font=("Helvetica", 10))
        self.label.grid(row=5, columnspan=2)

        # create a button for clearing all characters recorded
        self.button_clearall = tk.Button(self.window, text="Click to clear all", command=self.cmd_button_clearall)
        self.button_clearall.grid(row=6, columnspan=2)
        self.label = tk.Label(self.window, text=" ", font=("Helvetica", 5))
        self.label.grid(row=7, columnspan=2)

        # create a button for clearing all characters recorded
        self.button_undo = tk.Button(self.window, text="Click to undo", command=self.cmd_button_undo)
        self.button_undo.grid(row=8, columnspan=2)
        self.label = tk.Label(self.window, text=" ", font=("Helvetica", 10))
        self.label.grid(row=9, columnspan=2)

    # command function for clear-all button 
    def cmd_button_clearall(self):
        self.sentence.set(" ")
    
    # command function for undo button 
    def cmd_button_undo(self):
        text_tmp = self.sentence.get()
        if len(text_tmp) > 0:
            text_new = text_tmp[:-1]
            self.sentence.set(text_new)

    # function used to update GUI
    def GUI_update_gen1(self, img_left, img_right, sign_meaning='', sign_prob=''):
        
        # convert the frames from BGR to RGB
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # annotate the left video from webcam
        cv2.putText(img_left, sign_meaning, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1)
        cv2.putText(img_left, "prob: %.4f" % sign_prob, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1)

        # convert left frame to PIL Image format
        webcam_img = Image.fromarray(img_left)
        left_imgtk = ImageTk.PhotoImage(image=webcam_img)
        
        # update the left video frame in GUI
        self.left_frame.imgtk = left_imgtk
        self.left_frame.create_image(0, 0, image=left_imgtk, anchor=tk.NW)
        

        # resize the right frame
        img_right = cv2.resize(img_right, (360, 480))

        cv2.putText(img_right, sign_meaning, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        cv2.putText(img_right, "prob: %.4f" % sign_prob, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        
        # convert the small frame to PIL Image format
        hand_img = Image.fromarray(img_right)
        right_imgtk = ImageTk.PhotoImage(image=hand_img)
        
        # Update the right video frame in the UI
        self.right_frame.imgtk = right_imgtk
        self.right_frame.create_image(0, 0, image=right_imgtk, anchor=tk.NW)
        
        # Schedule the next update
        self.window.after(10, self.GUI_update_gen1)



    def GUI_update_gen2(self, img_left, img_right, sign_meaning='', sign_prob=''):
        
        # convert the frames from BGR to RGB
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # annotate the left video from webcam
        cv2.putText(img_left, sign_meaning, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1)
        cv2.putText(img_left, "prob: %.4f" % sign_prob, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1)

        # convert left frame to PIL Image format
        webcam_img = Image.fromarray(img_left)
        left_imgtk = ImageTk.PhotoImage(image=webcam_img)
        
        # update the left video frame in GUI
        self.left_frame.imgtk = left_imgtk
        self.left_frame.create_image(0, 0, image=left_imgtk, anchor=tk.NW)
        

        # resize the right frame
        img_right = cv2.resize(img_right, (360, 480))

        if sign_meaning == 'none_sign':

            cv2.putText(img_right, 'none', (100, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (255, 255, 255), 3)

        else:
            
            cv2.putText(img_right, sign2char_upper(sign_meaning), (140, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0,
                        (255, 255, 255), 3)

        # convert the small frame to PIL Image format
        hand_img = Image.fromarray(img_right)
        right_imgtk = ImageTk.PhotoImage(image=hand_img)
        
        # Update the right video frame in the UI
        self.right_frame.imgtk = right_imgtk
        self.right_frame.create_image(0, 0, image=right_imgtk, anchor=tk.NW)
        
        # Schedule the next update
        self.window.after(10, self.GUI_update_gen2)


    # function used to update the displayed sentence
    def update_sentence(self, new_char):
        new_char = new_char.replace('_',' ')
        text_tmp = self.sentence.get()
        text_new = text_tmp + new_char
        self.sentence.set(text_new)

# function used to transfer sign to character it indicates
def sign2char_lower(sign):
    if sign.startswith("sign_"):
        return chr(ord('a') + ord(sign[-1]) - ord('A'))
    elif sign.startswith("number_"):
        return sign[-1]
    elif sign.startswith("txt_"):
        return sign[4:]
    else:
        return None
    

def sign2char_upper(sign):
    if sign.startswith("sign_"):
        return chr(ord('A') + ord(sign[-1]) - ord('A'))
    elif sign.startswith("number_"):
        return sign[-1]
    elif sign.startswith("txt_"):
        return sign[4:]
    else:
        return None


# main function
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    
    # init model
    model = C3D_model.C3D(num_classes=NUM_CLASS)
    checkpoint = torch.load(MODEL_FILE, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # activate webcam and hand detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)

    # variables for classification
    label = 0
    previous_label = 0
    correct_label = 0
    label_list = []
    hand_point_img_list = []
    sign_meaning = 'none_sign'
    sign_meaning_pre = 'none_sign'

    # gui 
    my_GUI = GUI()

    # time variables
    time_cur = time.time()
    time_pre = time.time()

    # main loop
    while True:
        success, img_ori = cap.read()
        
        img_ori = cv2.flip(img_ori, 1)
        img_ori_copy = np.copy(img_ori)

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
        else:
            hand_point_img_cropped = np.zeros((300, 300, 3), np.uint8)


        if (len(hands) == 2) or (len(hands) == 1):

            # save video and do classification
            if len(hand_point_img_list) < FRAME_NUM:

                resize_tmp = cv2.resize(hand_point_img_cropped, (171, 128))
                tmp_ = center_crop(resize_tmp)
                tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                hand_point_img_list.append(tmp)
                
            else:
                hand_point_img_list.pop(0)
                resize_tmp = cv2.resize(hand_point_img_cropped, (171, 128))
                tmp_ = center_crop(resize_tmp)
                tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                hand_point_img_list.append(tmp)
                

            if len(hand_point_img_list) == FRAME_NUM:
                inputs = np.array(hand_point_img_list).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                inputs = torch.from_numpy(inputs)
                inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

                # input into model
                with torch.no_grad():
                    outputs = model.forward(inputs)

                probs = torch.nn.Softmax(dim=1)(outputs)
                label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

        # no hands case, there is no sign
        else:
            probs = torch.zeros((1, 37))
            probs[0][0] = 1.0 
            label = 0


        if previous_label == label:
            label_list.append(label)
            previous_label = label
        else:
            label_list = []
            previous_label = label
        
        # only if the a Auslan sign is found more than 5 times continuously, we say the sign is made
        if len(label_list) == 15:
            correct_label = label_list[0]
            label_list = []


        # annote in image
        with open(LABEL_FILE, 'r') as f:
            class_names = f.readlines()
            f.close()


        try:
            sign_meaning = class_names[correct_label].split(' ')[-1].strip()
            sign_prob = probs[0][correct_label]
        except:
            # print('error')
            pass
                    

        # GUI 
        # update the displayed sentence
        if sign_meaning == sign_meaning_pre and (sign_meaning != 'none_sign'):
            time_cur = time.time()
            sign_meaning_pre == sign_meaning
            time_pre = time_cur

        elif sign_meaning == 'none_sign':
            sign_meaning_pre = sign_meaning
            time_cur = time.time()
            if time_cur - time_pre > 4:
                my_GUI.update_sentence(' ')
                time_pre = time_cur
        else:
            time_cur = time.time()
            my_GUI.update_sentence(sign2char_lower(sign_meaning))
            sign_meaning_pre = sign_meaning
            time_pre = time_cur

        # update videos in GUI
        blank = np.zeros((300, 300, 3), np.uint8)
        if GUI_GEN == 1:
            my_GUI.GUI_update_gen1(img_ori_copy, hand_point_img_cropped, sign_meaning, sign_prob)
            my_GUI.window.update()

        elif GUI_GEN == 2:
            my_GUI.GUI_update_gen2(img_ori_copy, blank, sign_meaning, sign_prob)
            my_GUI.window.update()



if __name__ == '__main__':
    main()
