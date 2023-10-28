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

GUI_GEN = 2
FRAME_NUM = 40
NUM_CLASS = 44

# Auslan till sign_H
LABEL_FILE = 'Auslan_labels.txt'
MODEL_FILE = 'run\\run_1\\models\\C3D-Auslan_epoch-29.pth.tar'
CAMERA_HEIGHT = 500
CAMERA_WIDTH  = 650
label_cou = 0

# function used to extract the hand features
def feature_extraction(hands_data):
    feature_array = np.zeros((4, 21), dtype=np.float64)
    if (len(hands_data) != 1) and (len(hands_data) != 2):
        pass

    elif hands_data and len(hands_data) == 1:
        hand = hands_data[0]
        if hand['type'] == 'Left':  
            # only record hand landmarks
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
        self.label = tk.Label(self.window, text="hand feature array", font=("Helvetica", 15))
        self.label.grid(row=1, column=1)
        
        # display the video from webcam on the left
        self.left_frame = tk.Canvas(self.window, width=640, height=480)
        self.left_frame.grid(row=2, column=0)
        
        # display the cropped hand image on the right
        self.right_frame = tk.Canvas(self.window, width=525, height=100)
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
        # self.label.config(text="cleared!")

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
        img_right_uint8 = np.round(img_right).astype(np.uint8)


        # resize the right frame
        feature_array_255_large = cv2.resize(img_right_uint8, (525, 100))

        # convert the small frame to PIL Image format
        hand_data_array = Image.fromarray(feature_array_255_large)
        right_imgtk = ImageTk.PhotoImage(image=hand_data_array)

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
    elif sign.startswith("num_"):
        return sign[4:]
    elif sign.startswith("txt_"):
        return sign[4:]
    else:
        return None
    

def sign2char_upper(sign):
    if sign.startswith("sign_"):
        return chr(ord('A') + ord(sign[-1]) - ord('A'))
    elif sign.startswith("number_"):
        return sign[7:]
    elif sign.startswith("txt_"):
        return sign[4:]
    else:
        return None


# main function
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    
    # init model
    model = C3D_model.Linear3DModel(num_classes=NUM_CLASS)
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

        if len(hands) == 2:
            if hands[0]['type'] == hands[1]['type']:
                if hands[0]['center'][0] < hands[1]['center'][0]:
                    hands[0]['type'] = "Right"
                    hands[1]['type'] = "Left"
                else:
                    hands[0]['type'] = "Left"
                    hands[1]['type'] = "Right"
        
        feature_array = feature_extraction(hands)
        feature_array_255_1c = (feature_array * 255).astype(np.float16)
        feature_array_255 = np.zeros((4, 21, 3), dtype=np.float16)
        feature_array_255[:,:,0] = feature_array_255_1c
        feature_array_255[:,:,1] = feature_array_255_1c
        feature_array_255[:,:,2] = feature_array_255_1c
        
        if (len(hands) == 2) or (len(hands) == 1):
            
            # save video and do classification
            if len(hand_point_img_list) < FRAME_NUM:
                tmp = feature_array_255 - np.array([[[100.0, 100.0, 100.0]]])
                hand_point_img_list.append(tmp)

                
            else:
                hand_point_img_list.pop(0)
                tmp = feature_array_255 - np.array([[[100.0, 100.0, 100.0]]])
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
            probs = torch.zeros((1, NUM_CLASS))
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
        else:
            pass


        # annote in image
        with open(LABEL_FILE, 'r') as f:
            class_names = f.readlines()
            f.close()
        
        sign_meaning = class_names[correct_label].split(' ')[-1].strip()
        sign_prob = probs[0][correct_label]


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
            my_GUI.GUI_update_gen1(img_ori_copy, img_ori_copy, sign_meaning, sign_prob)
            my_GUI.window.update()

        elif GUI_GEN == 2:
            my_GUI.GUI_update_gen2(img_ori_copy, feature_array_255_1c, sign_meaning, sign_prob)
            my_GUI.window.update()



if __name__ == '__main__':
    main()
