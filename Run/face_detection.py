'''
This is the file for performing simple face detection:
    1. train corresponding classfier
    2. bounded box proposal 
    3. Merge bounded box
    4. Draw bounded box 
'''
import sys, os
sys.path.append("./")
from Utils.DataLoader import DataLoader
from Models.LogisticModel import LogisticRegression
from Models.FisherModel import FisherModel
from Models.SVM_Model import SVM
from Models.CNN_Model import VanillaCNN
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np
import cv2

Data = DataLoader()
Data.load_pickle_dataset()

def extract_hog_feature(image):
    hog_feature = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return hog_feature

def generate_proposals(width, height):
    '''
    One proposal is [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)]
    '''
    proposals = []
    for size in range(10, 100):
        for i in range(width - size):
            for j in range(height - size):
                top_left = (i, j)
                bot_right = (i+size, j+size)
                proposals.append([top_left, bot_right])
    return proposals

def draw_bounded_box(img, bounded_boxes):
    for (top_left, bot_right) in bounded_boxes:
        img = cv2.rectangle(img,top_left,bot_right,(255,0,0),1)
    return img

def face_detection(image, classfier = ""):
    rgr_model = LogisticRegression()
    rgr_model.fit_RGD(Data.train_data, Data.train_label)
    rgr_model.test_acc(Data.test_data, Data.test_label)

    orig_width = image.shape[1]
    orig_height = image.shape[0]

    proposals = generate_proposals(orig_width, orig_height)

    bounded_boxes = []
    for tmp_bounded_box in proposals:
        top_left = tmp_bounded_box[0]
        bot_right = tmp_bounded_box[1]
        tmp_img = image[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]

        if rgr_model.predict(tmp_img):
            bounded_boxes.append(tmp_bounded_box)

    detected_image = draw_bounded_box(image, bounded_boxes)

    # ended
    plt.imshow(detected_image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

if __name__ == "__main__":
    file_name = "./2003/01/01/big/img_230.jpg"
    image = cv2.imread(file_name)
    b, g, r = cv2.split(image)
    image = cv2.merge([r,g,b])

    rgr_model = LogisticRegression()
    rgr_model.fit_RGD(Data.train_data, Data.train_label)
    rgr_model.test_acc(Data.test_data, Data.test_label)

    orig_width = image.shape[1]
    orig_height = image.shape[0]

    proposals = generate_proposals(orig_width, orig_height)

    bounded_boxes = []
    for tmp_bounded_box in proposals:
        top_left = tmp_bounded_box[0]
        bot_right = tmp_bounded_box[1]
        tmp_img = image[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
        tmp_img = cv2.resize(tmp_img, (96, 96))


        tmp_hog = [extract_hog_feature(tmp_img)]
        tmp_hog = np.array(tmp_hog)

        if rgr_model.predict(tmp_hog):
            bounded_boxes.append(tmp_bounded_box)

    detected_image = draw_bounded_box(image, bounded_boxes)

    # ended
    plt.imshow(detected_image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show() 
    