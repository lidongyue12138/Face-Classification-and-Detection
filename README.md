# Project 1: Face Classification and Detection

This is an implementation for the first project of *CS385 Machine Learning* at Shanghai Jiao Tong University, instructed by Prof. Quanshi Zhang and Dr. Xu Cheng. I implemented some linear, kernel-based models as well as CNNs to do face classification, and also a sliding-window-based face detector.

### Important Dependencies

- python 3.5+
- numpy 
- matplotlib
- skimage
- python-opencv
- tensorflow 

### Dataset

I use Face Detection Data Set and Benchmark ([FDDB](http://vis-www.cs.umass.edu/fddb/)) in this project. To run the code, you have to download the dataset and its annotations. Extract the dataset in the root directory.

The whole directory of this project should be like:

```
-- 2002

-- 2003

-- FDDB-folds

-- FDDB-folds-ellipse

-- Models

------ CNN_Model.py

------ FisherModel.py

------ LogisticModel.py

------ SVM_Model.py

-- Run 

------- run.py

------- face_detection.py

-- Utils

------ DataLoader.py

------ feature_visualization.py

------ visualize_HOG.py
```

#### Run the code

To run the code, fist you have to load the data:

```python
from Utils.DataLoader import DataLoader

DataLoader = DataLoader()
DataLoader.load_dataset_new()
DataLoader.load_pickle_dataset_new()
```

To run the classifier, you should refer to `run.py`  to see the examples:

```python
import sys, os
sys.path.append("./")
from Utils.DataLoader import DataLoader
from Models.LogisticModel import LogisticRegression
from Models.FisherModel import FisherModel
from Models.SVM_Model import SVM
from Models.CNN_Model import VanillaCNN
import numpy as np
import matplotlib.pyplot as plt

Data = DataLoader()
Data.load_pickle_dataset_new()

''' CNN Model '''
cnn = VanillaCNN()

for i in range(100):
    batch_X, batch_y = Data.next_batch_train(32)
    loss, acc = cnn.train_model(batch_X, batch_y)

    if i%10 == 0:
        print("Training loss: %f\t accuracy: %f" %(loss, acc))

for i in range(10):
    batch_X, batch_y = Data.next_batch_test(200)
    cnn.test_acc(batch_X, batch_y)

''' SVM '''
model = SVM()
model.train_and_test(
    Data.train_data_hog, Data.train_label,
    Data.test_data_hog, Data.test_label
)
from sklearn.svm import SVC
clf = SVC(gamma='auto', kernel="rbf", class_weight='balanced')
clf.fit(Data.train_data_hog, Data.train_label) 


''' Fisher Model '''
fisher_model = FisherModel()
fisher_model.fit_gaussian(Data.train_data_hog, Data.train_label)
fisher_model.test_acc_gaussian(Data.test_data_hog, Data.test_label)

print(np.dot((fisher_model.mean_one - fisher_model.mean_zero).T, fisher_model.w)**2)
within = fisher_model.num_one * np.dot(fisher_model.w.T, np.dot(fisher_model.cov_one, fisher_model.w)) + \
    fisher_model.num_zero * np.dot(fisher_model.w.T, np.dot(fisher_model.cov_zero, fisher_model.w))
print(within)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# clf = LinearDiscriminantAnalysis()
# clf.fit(Data.train_data_hog, Data.train_label)
# print(clf.score(Data.test_data_hog, Data.test_label))

''' Logistic Model '''
rgr_model = LogisticRegression()
rgr_model.fit_langevin_dynamics(Data.train_data_hog, Data.train_label)
rgr_model.test_acc(Data.test_data_hog, Data.test_label)

''' Sklearn Logistic Regression'''
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state=0, solver='lbfgs',
#                          multi_class='multinomial').fit(Data.train_data_hog, Data.train_label)
# clf.score(Data.test_data_hog, Data.test_label)

''' Visualize support vector '''

# support_index = [2, 25, 30, 32, 38, 41]
# images = [Data.train_data_img[i] for i in support_index]

# plt.figure() #设置窗口大小
# # plt.suptitle('') # 图片名称
# for i in range(6):
#     plt.subplot(2,3,i+1), plt.title('')
#     plt.imshow(images[i]), plt.axis('off')
# plt.show()
```

To run face detection, you should refer to `face_detection.py`:

```python
from sklearn.linear_model import LogisticRegression
rgr_model = LogisticRegression(random_state=0, solver='lbfgs', 			       multi_class='multinomial').fit(Data.train_data_hog, Data.train_label)

file_name = "./2002/07/31/big/img_593.jpg"

face_detection(rgr_model, file_name)
```

### Contact 

If you have any problem about this code, please contact me through: lidongyue AT sjtu DOT edu DOT cn

## Deprecated

### TO DO

1. Face Classification
   - Data Processing
     - Generate positive samples: by the annotation of the dataset
     - Generate negative samples: by resizing or by sliding windows
     - Extract HOG features
     - Generate training set and testing set
   - Model Building
     - Logistic Model
       - SGD
       - Langevin dynamics
     - Fisher Model
     - SVM Model
     - CNN Model
   - Visualization
     - Visualize bounding boxes of positive and negative samples on images. 
     - Visualize the extracted HOG features 
     - Visualize face detection results and feature distribution 
2. Face Detection

### Tasks

- [x] Visualize bounding boxes of positive and negative samples on images. 
- [x] Visualize the extracted HOG features 
- [x] Learning a logistic model for classification: SGD & Langevin
  - [ ]   Report the accuracy 
- [ ] Learning a fisher model for classification 
  - [ ] Report the accuracy, the intra-class variance and the inter-class variance 
- [ ] Learning SVMs for classification 
  - [ ] Report the accuracy 
  - [ ] List samples of support vectors (i.e., the samples whose the margin =1) 
- [ ] Learning convolutional neural networks for classification 
- [ ] Visualize face detection results and feature distribution 

### ToDo

- resplit training and testing set
- fix each model performance
- create api for face detection in each model: predict()

### Results

|           Model           | Training  Accuracy | Testing Accuracy |
| :-----------------------: | :----------------: | :--------------: |
|   Logistic Model (RGD)    |       96.04        |      96.18       |
| Logistic Model (Langevin) |       91.09        |      90.22       |
|       SVM (linear)        |                    |      92.41       |
|         SVM (rbf)         |                    |      96.39       |
|       Fisher Model        |                    |      89.49       |
|            CNN            |                    |      94.10       |
| Sklearn - Logistic Model  |         \          |      97.17       |
|       Sklearn - LDA       |         \          |      96.97       |
|                           |                    |                  |
