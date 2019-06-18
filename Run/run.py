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