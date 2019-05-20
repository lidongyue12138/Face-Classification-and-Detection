'''
This is the data loader file, it is supposed to do the following
TODO:
    1. Load each image, find the face, generate positive samples (done)
    2. Generate negative samples (done)
    3. Extract HOG feature (done)
    4. Generate training set and testing set (done)

    5. Visualize the extracted HOG features
'''

# Import Information
from matplotlib import pyplot as plt
from skimage.feature import hog
import numpy as np
import pickle
import os
import cv2

# Directory Information
CUR_DIR = "./"
FOLDS_DIR = "./FDDB-folds/"
ELLIPSE_DIR = "./FDDB-folds-ellipse/"

TARGET_SIZE = (96, 96) # all image resized to this size

# Data Loader Definition
class DataLoader:
    def __init__(self):
        self.fold_list = os.listdir(FOLDS_DIR)
        self.ellipse_list = os.listdir(ELLIPSE_DIR)

        # self.train_folds = self.fold_list[:7]
        # self.train_ellipses = self.ellipse_list[:7]
        # self.test_folds = self.fold_list[8:]
        # self.test_ellipses = self.ellipse_list[8:]

        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

    # ----------------------------------------- CUTE SPLIT LINE -------------------------------------------------
    '''
    Only load image data into dataset:
        1. Find positive examples
        2. Generate negative samples
        3. Shuffle the whole dataset
    '''
    def load_dataset(self):
        self.whole_data = []
        self.whole_label = []

        ''' Generate positive samples for training set '''
        for i, ellipse_fold in enumerate(self.ellipse_list):
            for face_img in self.generate_clipped_faces(ellipse_fold):
                tmp_hog_feature = self.extract_hog_feature(face_img)
                self.whole_data.append(tmp_hog_feature)
                self.whole_label.append(1)
            print("========== Loading positive samples in NO.%d fold finished =========" %(i+1))
        ''' Ended '''

        ''' Generate negative samples for training set '''
        for i, fold in enumerate(self.fold_list):
            for negative_img in self.generate_negative_samples(fold, mode = "ORIGINAL"):
                tmp_hog_feature = self.extract_hog_feature(negative_img)
                self.whole_data.append(tmp_hog_feature)
                self.whole_label.append(0)
            print("========== Loading negative samples in NO.%d fold finished =========" %(i+1))
        ''' Ended '''
        
        ''' Shuffle data '''
        self.whole_data = np.array(self.whole_data)
        self.whole_label = np.array(self.whole_label)
        permutation = np.random.permutation(len(self.whole_data))
        self.whole_data[permutation]
        self.whole_label[permutation]
        ''' Ended '''

        print("========== Loading all fold finished =========")
        print("whole data size: (%d,\t %d)" %(self.whole_data.shape[0], self.whole_data.shape[1]))
        
        self.train_data = self.whole_data[:int(len(self.whole_data)*0.8)]
        self.train_label = self.whole_label[:int(len(self.whole_data)*0.8)]
        self.test_data = self.whole_data[int(len(self.whole_data)*0.8)+1:]
        self.test_label = self.whole_label[int(len(self.whole_data)*0.8)+1:]

        print("========== Loading training and testing finished =========")


    # ----------------------------------------- CUTE SPLIT LINE -------------------------------------------------
    def read_images_names(self, file_name):
        fr = open(os.path.join(FOLDS_DIR, file_name), "r")
        for img_dir in fr.readlines():
            img_dir = img_dir.strip() + ".jpg"
            yield img_dir

    '''
    This is supposed to read one image information at one time
    '''
    def read_images_info(self, ellipse_file_name):
        fr = open(os.path.join(ELLIPSE_DIR, ellipse_file_name), "r")
        while True:
            img_dir = fr.readline()
            img_dir = img_dir.strip() + ".jpg"
            if not img_dir:
                break

            try:
                face_num = int(fr.readline())
                for _ in range(face_num):
                    tmp_info_line = fr.readline()
                    tmp_info_line = tmp_info_line.split()
                    
                    major_axis_radius = float(tmp_info_line[0])
                    minor_axis_radius = float(tmp_info_line[1])
                    angle = float(tmp_info_line[2])
                    center_x = float(tmp_info_line[3])
                    center_y = float(tmp_info_line[4])
                    yield img_dir, major_axis_radius, minor_axis_radius, angle, center_x, center_y
            except:
                break

    # ----------------------------------------- CUTE SPLIT LINE -------------------------------------------------

    # ----------------------------------------- CUTE SPLIT LINE -------------------------------------------------
    '''
    This is supposed to return all negative samples
        1. There is two mode to generate negative samples: 
            "ORIGINAL": directly resize original images into 96 x 96 pixels
            "SLIDING": generate eight negative images based on each face by sliding the
                       bounding box by 1/3
        2. Input:
            ellipse_file_name
           Output:
            generator of negative samples
    '''
    def generate_negative_samples(self, file_name, mode = "ORIGINAL"):
        if mode == "ORIGINAL":
            for img_dir in self.read_images_names(file_name):
                img_dir = os.path.join(CUR_DIR, img_dir)
                img = cv2.imread(img_dir)

                orig_width = img.shape[1]
                orig_height = img.shape[0]

                ''' if the color reformation is needed '''
                b, g, r = cv2.split(img)
                img = cv2.merge([r,g,b])
                ''' ended '''

                resized_img = cv2.resize(img, TARGET_SIZE, cv2.INTER_AREA)
                yield resized_img
        if mode == "SLIDING":
            for info_line in self.read_images_info(file_name):
                img_dir, major_axis_radius, minor_axis_radius, angle, center_x, center_y = info_line

                major_axis_radius = int(major_axis_radius)
                minor_axis_radius = int(minor_axis_radius)
                angle = int(angle)
                center_x = int(center_x)
                center_y = int(center_y)

                img_dir = os.path.join(CUR_DIR, img_dir)
                img = cv2.imread(img_dir)

                orig_width = img.shape[1]
                orig_height = img.shape[0]

                ''' if the color reformation is needed '''
                b,g,r=cv2.split(img)
                img=cv2.merge([r,g,b])
                ''' ended '''

                ''' rectangle box information '''
                face_width = int(minor_axis_radius)
                face_height = int(major_axis_radius)
                top_left = (center_x - face_width, center_y - face_height)
                bottom_right = (center_x + face_width, center_y + face_height)
                ''' ended '''

                '''
                Sliding Window List:
                    (sliding_left, sliding_up): 
                        1 means sliding to the direction with 1/3*width or 1/3*height
                        0 means no sliding
                        -1 means sliding to the opposite direction
                '''
                sliding_window = [(1, 0), (1, 1), (0, 1), (-1, 1),
                                (-1,0), (-1, -1), (0, -1), (1, -1)]

                for i in range(8):
                    slide_left, slide_up = sliding_window[i]
                    slide_left = int(slide_left * face_width * (2/3))
                    slide_up = int(slide_up * face_height * (2/3))

                    subimg_top_left_y = top_left[1] - slide_up if top_left[1] - slide_up > 0 else 0
                    subimg_top_left_x = top_left[0] - slide_left if top_left[0] - slide_left > 0 else 0
                    subimg_bot_right_y = bottom_right[1] - slide_up if bottom_right[1] - slide_up < orig_height else orig_height
                    subimg_bot_right_x = bottom_right[0] - slide_left if bottom_right[0] - slide_left < orig_width else orig_width
                    slided_img = img[subimg_top_left_y:subimg_bot_right_y, subimg_top_left_x:subimg_bot_right_x]
                    
                    slided_img = cv2.resize(slided_img, TARGET_SIZE, interpolation = cv2.INTER_AREA)
                    yield slided_img
                
    '''
    This is supposed to return all positive face images
        Input: ellipse_file_name
        Output: generator of faces data
    '''
    def generate_clipped_faces(self, ellipse_file_name):
        for info_line in self.read_images_info(ellipse_file_name):
            ''' every denoted face information '''
            img_dir, major_axis_radius, minor_axis_radius, angle, center_x, center_y = info_line

            major_axis_radius = int(major_axis_radius)
            minor_axis_radius = int(minor_axis_radius)
            angle = int(angle)
            center_x = int(center_x)
            center_y = int(center_y)
            ''' ended '''

            ''' read corresponding face image '''
            img_dir = os.path.join(CUR_DIR, img_dir)
            img = cv2.imread(img_dir)

            orig_width = img.shape[1]
            orig_height = img.shape[0]
            ''' ended '''

            ''' if the color reformation is needed '''
            b,g,r=cv2.split(img)
            img=cv2.merge([r,g,b])
            ''' ended '''

            ''' rectangle box information '''
            face_width = int(minor_axis_radius*(4/3))
            face_height = int(major_axis_radius*(4/3))
            top_left = (center_x - face_width, center_y - face_height)
            bottom_right = (center_x + face_width, center_y + face_height)
            ''' ended '''
            
            ''' substract a region of picture '''
            subimg_top_left_y = top_left[1] if top_left[1] > 0 else 0
            subimg_top_left_x = top_left[0] if top_left[0] > 0 else 0
            subimg_bot_right_y = bottom_right[1] if bottom_right[1] < orig_height else orig_height
            subimg_bot_right_x = bottom_right[0] if bottom_right[0] < orig_width else orig_width
            face_img = img[subimg_top_left_y:subimg_bot_right_y, subimg_top_left_x:subimg_bot_right_x]
            
            '''
            If you want to fill in the empty space
            '''
            # face_img = cv2.copyMakeBorder(
            #     face_img,
            #     top=abs(subimg_top_left_y - top_left[1]),
            #     bottom=abs(subimg_bot_right_y - bottom_right[1]),
            #     left=abs(subimg_top_left_x - top_left[0]),
            #     right=abs(subimg_bot_right_x - bottom_right[0]),
            #     borderType=cv2.BORDER_REPLICATE
            # )
            ''' ended '''

            ''' show the face image '''
            # plt.imshow(face_img, cmap = 'gray', interpolation = 'bicubic')
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()
            ''' ended '''

            ''' resize image '''
            face_img = cv2.resize(face_img, TARGET_SIZE, interpolation = cv2.INTER_AREA)
            ''' ended '''

            yield face_img
    # ----------------------------------------- CUTE SPLIT LINE -------------------------------------------------

    # ----------------------------------------- CUTE SPLIT LINE -------------------------------------------------
    '''
    This is used to visualize the ellipse and rectangle window of each face
    '''
    def visualize_face(self, file_name):
        count = 0
        for info_line in self.read_images_info(tmp_file):
            img_dir, major_axis_radius, minor_axis_radius, angle, center_x, center_y = info_line
            print(img_dir)
            print(major_axis_radius)
            print(minor_axis_radius)
            print(angle)
            print(center_x)
            print(center_y)

            major_axis_radius = int(major_axis_radius)
            minor_axis_radius = int(minor_axis_radius)
            angle = int(angle)
            center_x = int(center_x)
            center_y = int(center_y)

            img_dir = os.path.join(CUR_DIR, img_dir)
            img = cv2.imread(img_dir)

            # draw the ellipse face circcle
            cv2.ellipse(
                img=img,
                center=(center_x, center_y),
                axes=(minor_axis_radius, major_axis_radius),
                angle=angle,
                startAngle=0, endAngle=360,
                color=(0,0,255),
                thickness=3
            )
            # ended

            # draw the rectangle box
            face_width = int(minor_axis_radius*(4/3))
            face_height = int(major_axis_radius*(4/3))
            top_left = (center_x - face_width, center_y - face_height)
            bottom_right = (center_x + face_width, center_y + face_height)

            img = cv2.rectangle(img,top_left,bottom_right,(255,0,0),3)
            # ended

            # if the color reformation is needed
            b,g,r=cv2.split(img)
            img=cv2.merge([r,g,b])
            # ended
            plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()
            
            count += 1
            if count > 20:
                break 

    # ----------------------------------------- CUTE SPLIT LINE -------------------------------------------------
    def extract_hog_feature(self, image):
        hog_feature = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
        return hog_feature
    # ----------------------------------------- CUTE SPLIT LINE -------------------------------------------------
    def load_pickle_dataset(self):
        fr_train_data = open("./train_data.pkl", 'rb')
        fr_train_label = open("./train_label.pkl", 'rb')
        fr_test_data = open("./test_data.pkl", 'rb')
        fr_test_label = open("./test_label.pkl", 'rb')

        self.train_data = pickle.load(fr_train_data)
        self.train_label = pickle.load(fr_train_label)
        self.test_data = pickle.load(fr_test_data)
        self.test_label = pickle.load(fr_test_label)
        
        fr_train_data.close()
        fr_train_label.close()
        fr_test_data.close()
        fr_test_label.close()

        print("========== Loading training and testing finished =========")
    
    def save_dataset(self):
        self.save_to_pickle(self.train_data, "train_data")
        self.save_to_pickle(self.train_label, "train_label")
        self.save_to_pickle(self.test_data, "test_data")
        self.save_to_pickle(self.test_label, "test_label")

    def save_to_pickle(self, data, name):
        fw = open(name + ".pkl", 'wb')
        pickle.dump(data, fw)
        fw.close()

if __name__ == "__main__":
    DataLoader = DataLoader()
    # DataLoader.load_dataset()
    # DataLoader.save_dataset()
    DataLoader.load_pickle_dataset()