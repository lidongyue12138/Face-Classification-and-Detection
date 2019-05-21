import sys, os
sys.path.append("./")
from Utils.DataLoader import DataLoader
from Models.LogisticModel import LogisticRegression
from Models.FisherModel import FisherModel
from Models.SVM_Model import SVM

Data = DataLoader()
Data.load_pickle_dataset()

''' SVM '''
# model = SVM()
# model.train_and_test(
#     Data.train_data, Data.train_label,
#     Data.test_data, Data.test_label
# )

''' Fisher Model '''
fisher_model = FisherModel()
fisher_model.fit_gaussian(Data.train_data, Data.train_label)
fisher_model.test_acc_gaussian(Data.test_data, Data.test_label)

''' Logistic Model '''
# rgr_model = LogisticRegression()
# rgr_model.fit_langevin_dynamics(Data.train_data, Data.train_label)