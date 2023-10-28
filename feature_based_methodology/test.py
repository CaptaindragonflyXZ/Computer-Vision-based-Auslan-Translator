import os
import torch
import numpy as np
from network import C3D_model
import cv2
import shutil
from sklearn.model_selection import train_test_split
torch.backends.cudnn.benchmark = True
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt


RECORD_BAD_FILES = False

# Auslan till sign_H
LABEL_FILE = 'Auslan_labels.txt'
MODEL_FILE = 'run\\run_4\\models\\C3D-Auslan_epoch-29.pth.tar'


# xina test
TEST_FOLDER = "Data_processed\\test"
SAVE_FOLDER = 'test_result\\Xina'


# # zichen
# TEST_FOLDER = "ZICHEN_test_set\\test"
# SAVE_FOLDER = 'test_result\\Zichen'


# # yupeng
# TEST_FOLDER = "YUPENG_test_set\\test"
# SAVE_FOLDER = 'test_result\\Yupeng'   


PROCESS_VIDEO = True
NUM_CLASS = 44


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


# read jpg files in a given path
def read_jpg_files_in_folder(folder_path):
    jpg_files = []

    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        raise ValueError("The provided path is not a directory.")

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a JPG file
        if filename.lower().endswith(".jpg") and len(jpg_files) < 40:
            try:
                # Open and read the JPG file using PIL
                img = Image.open(file_path)
                img_array = np.array(img)
                tmp = img_array - np.array([[[100.0, 100.0, 100.0]]])
                jpg_files.append(tmp)
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")

    return jpg_files





def main():
    print('MODEL_FILE is',MODEL_FILE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open(LABEL_FILE, 'r') as f:
        class_names = f.readlines()
        f.close()


    # init model
    model = C3D_model.Linear3DModel(num_classes=NUM_CLASS)
    checkpoint = torch.load(MODEL_FILE, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    
    TP = 0  # true positive,  when correctly classify Auslan as Auslan with correct meaning
    TN = 0  # true negative,  when correctly identifies non-sign languages as non-sign languages
    FP = 0  # false positive, when misidentifying non-sign language as Auslan
    FN = 0  # false negative, when wrongly classify Auslan as Auslan with incorrect meaning

    true_labels = []
    predicted_labels = []
    labels_list = []


    subdirectories = [subdir for subdir in os.listdir(TEST_FOLDER) if os.path.isdir(os.path.join(TEST_FOLDER, subdir))]
    for i in subdirectories:
        labels_list.append(i)

    # Iterate through each subdirectory
    for this_sign in subdirectories:
        correct_label = this_sign

        TP_local = 0  
        TN_local = 0  
        FP_local = 0  
        FN_local = 0  

        subfolder_path = os.path.join(TEST_FOLDER, this_sign)
        all_samples = [file for file in os.listdir(subfolder_path)]

        # Display the .avi files one by one and store the folder name
        for this_sample in all_samples:
            
            sample_path = os.path.join(subfolder_path, this_sample)

            hand_data_queue = read_jpg_files_in_folder(sample_path)

            inputs = np.array(hand_data_queue).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)

            # input into model
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label_order = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            predict_label = class_names[label_order].split(' ')[-1].strip()
            
            if predict_label == 'num_10':
                predict_label = 'num_ten'

            # print('predicted result is', predict_label)
            true_labels.append(correct_label)
            predicted_labels.append(predict_label)

            if predict_label == correct_label:
                if (predict_label != 'none_sign'):
                    TP_local += 1
                else:
                    TN_local += 1
            else: 
                if (correct_label == 'none_sign'):
                    FP_local += 1
                else:
                    FN_local += 1

        TP += TP_local
        TN += TN_local
        FP += FP_local
        FN += FN_local

        local_accuracy = (TP_local+TN_local)/(TP_local+TN_local+FP_local+FN_local)
        print(f'{correct_label} accuracy is', local_accuracy)

    print('====== overall test resutls ======')
    overall_accuracy = (TP+TN)/(TP+TN+FP+FN)
    overall_recall = (TP)/(TP+FN)
    overall_precise = (TP)/(TP+FP)
    overall_F1_SCORE = (2*overall_precise*overall_recall)/(overall_precise+overall_recall)
    print('overall accuracy is', overall_accuracy)
    print('overall recall is', overall_recall)
    print('overall precise is', overall_precise)
    print('overall F1 score is', overall_F1_SCORE)
    plot_confusion_matrix(true_labels, predicted_labels, labels=labels_list)



def display_avi_files(folder_path):
    # Get a list of all subdirectories within the folder
    subdirectories = [subdir for subdir in os.listdir(TEST_FOLDER) if os.path.isdir(os.path.join(TEST_FOLDER, subdir))]

    # Iterate through each subdirectory
    for subdir in subdirectories:
        subfolder_path = os.path.join(TEST_FOLDER, subdir)
        avi_files = [file for file in os.listdir(subfolder_path) if file.endswith(".avi")]

        # Display the .avi files one by one and store the folder name
        for file in avi_files:
            file_path = os.path.join(subfolder_path, file)
            folder_name = subdir
            correct_label = folder_name
            print("File:", file_path)
            print("Folder Name:", folder_name)
            print()  # Print an empty line for separation



# function used to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels=None):

    # Check if labels are specified
    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create a heatmap plot
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig(f'{SAVE_FOLDER}\\epoch num VS test accuracy.png')




def process_video(video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(TEST_FOLDER, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 50 frames
        EXTRACT_FREQUENCY = 1
        frame_num = 40
        if frame_count // EXTRACT_FREQUENCY <= frame_num:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= frame_num:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= frame_num:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()


if __name__ == '__main__':
    main()