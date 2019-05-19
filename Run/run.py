from Utils.DataLoader import DataLoader
from Models.LogisticModel import LogisticRegression

Data = DataLoader()
Data.load_dataset()

rgr_model = LogisticRegression()
rgr_model.fit_langevin_dynamics(Data.train_data, Data.train_label)