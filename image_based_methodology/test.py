import os
import torch
import numpy as np
from network import C3D_model
import cv2
import shutil
torch.backends.cudnn.benchmark = True
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt


NUM_CLASS = 44
LABEL_FILE  = 'Auslan_labels_44.txt'
TEST_FOLDER = "ZICHEN_test_set_cnn3d\\test"
TEST_FOLDER = "YUPENG_test_set_cnn3d\\test"
MODEL_FILE  = 'run\\run_5\\models\\C3D-Auslan_epoch-59.pth.tar'



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



# read jpg files in a given path
def read_jpg_files_in_folder(folder_path):
    jpg_files = []

    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        raise ValueError("The provided path is not a directory.")

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # print('filename is', filename)
        # Check if the file is a JPG file
        if filename.lower().endswith(".jpg") and len(jpg_files) < 16:
            try:
                # Open and read the JPG file using PIL
                img = cv2.imread(file_path)
                img_array = img

                resize_tmp = cv2.resize(img_array, (171, 128))
                tmp_ = center_crop(resize_tmp)
                tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                jpg_files.append(tmp)

            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")

    return jpg_files



def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)




def main():
    print('MODEL_FILE is',MODEL_FILE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open(LABEL_FILE, 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    model = C3D_model.C3D(num_classes=NUM_CLASS)
    checkpoint = torch.load(MODEL_FILE, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    TP = 0  # true positive,  when correctly classifiy Auslan as Auslan with correct meanning
    TN = 0  # true negative,  when correctly identifies non-sign languages as non-sign languages
    FP = 0  # false positive, when misidentifying non-sign language as Auslan
    FN = 0  # false negative, when wrongly classifiy Auslan as Auslan with incorrect meanning

    true_labels = []
    predicted_labels = []
    labels_list = []

    subdirectories = [subdir for subdir in os.listdir(TEST_FOLDER) if os.path.isdir(os.path.join(TEST_FOLDER, subdir))]
    for i in subdirectories:
        labels_list.append(i)

    # Iterate through each subdirectory
    for this_sign in subdirectories:

        correct_label = this_sign
        # print('correct_label is',correct_label)

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
    

if __name__ == '__main__':
    main()