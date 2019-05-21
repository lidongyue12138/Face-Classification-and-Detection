'''
This is the code for Model Construction: SVM Model
'''
# necessary imports
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

class SVM:
    def __init__(self, kernel="rbf"):
        self.kernel = kernel

    def train_and_test(self, train_X, train_y, test_X, test_y):
        print('Begin to train\nX dimention:\t{}'.format(train_X.shape[1]))
        svc = SVC(kernel=self.kernel, class_weight='balanced', )
        
        c_range = np.logspace(-2, 10, 4, base=2)
        gamma_range = np.logspace(-5, 3, 5, base=2)
        param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
        grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=4)
        
        clf = grid.fit(train_X, train_y)
        test_acc = grid.score(test_X, test_y)
        
        print("Testing accuracy: %f" %test_acc)
