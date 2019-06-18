'''
This is the code for Model Construction: Fisher Model
    1. project X to one-dimension: calculate beta
    2. Use beta to calculate
'''
# Necessary Imports
import numpy as np

class FisherModel:
    def __init__(self):
        pass

    '''
    This Fisher model only consider our special case: 
        only for two classes
        only for classification
    '''
    def estimate_params(self, X, y):
        ''' 
        1. Calculate means for each classes
        2. Calculate the overall mean
        3. Calculate between class covariance matrix
        4. Calculate within class covariance matrix
        5. Find eigenvalue, eigenvector pairs for inv(S_W).S_B
        6. Sort the eigvals in decreasing order
        7. Take the first num_dims eigvectors
        '''
        # Calculate means for each classes
        self.mean_zero = np.mean(X[y==0], axis = 0)
        cov_zero = np.cov(X[y==0], rowvar=False)
        num_zero = len(X[y==0])
        self.mean_one = np.mean(X[y==1], axis = 0)
        cov_one = np.cov(X[y==1], rowvar=False)
        num_one = len(X[y==1])

        self.cov_zero = cov_zero
        self.num_zero = num_zero
        self.cov_one = cov_one
        self.num_one = num_one

        # Calculate the overall mean
        self.overall_mean = np.mean(X, axis = 0)

        # Calculate between class covariance matrix
        S_B = np.dot(self.mean_zero - self.mean_one, (self.mean_zero - self.mean_one).T) 

        # Calculate within class covariance matrix
        S_W = num_zero*cov_zero + num_one*cov_one

        '''
        This is equation given by Quanshi Zhang
        '''
        self.w = np.dot(np.linalg.pinv(S_W), self.mean_zero - self.mean_one)
        self.num_dims = 1
        
        # '''
        # This is by calculating eigenvalue
        # To be fixed
        # '''
        # # Find eigenvalue, eigenvector pairs for inv(S_W).S_B
        # mat = np.dot(np.linalg.pinv(S_W), S_B)
        # eigvals, eigvecs = np.linalg.eig(mat)
        # eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

        # # Sort the eigvals in decreasing order
        # eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)

        # # Take the first num_dims eigvectors
        # self.num_dims = 1
        # w = np.array([eiglist[i][1] for i in range(self.num_dims)])
        # self.w = w.T
        # self.w = np.reshape(self.w, (900,))

        # print(self.w.shape)
        
    '''
    Function to calculate the classification threshold.
    Projects the means of the classes and takes their mean as the threshold.
    Also specifies whether values greater than the threshold fall into class 1 or class 2.
    '''
    def fit_threshold(self, X, y):
        self.estimate_params(X, y)

        self.threshold = np.dot(self.overall_mean, self.w)
        self.value_first = np.dot(self.mean_zero, self.w)

        proj_X = np.dot(X, self.w)
        
        y_pred = np.zeros(y.size)
        y_pred[(proj_X>self.threshold) != (self.value_first>self.threshold)] = 1

        train_acc = np.mean(y_pred == y)
        print("Training accuracy: %f" %train_acc)


    def test_acc_threshold(self, X, y):
        proj_X = np.dot(X, self.w)
        
        y_pred = np.zeros(y.size)
        y_pred[(proj_X>self.threshold) != (self.value_first>self.threshold)] = 1

        test_acc = np.mean(y_pred == y)
        print("Testing accuracy: %f" %test_acc)

    def fit_gaussian(self, X, y):
        self.estimate_params(X, y)
        
        # Calculate the gaussian parameters
        proj_zero = np.dot(X[y==0], self.w)
        self.prior_zero = X[y==0].shape[0]/X.shape[0]
        self.gaussian_mean_zero = np.mean(proj_zero, axis=0)
        self.gaussian_cov_zero = np.cov(proj_zero, rowvar=False) 

        proj_one = np.dot(X[y==1], self.w)
        self.prior_one = X[y==1].shape[0]/X.shape[0]
        self.gaussian_mean_one = np.mean(proj_one, axis=0)
        self.gaussian_cov_one = np.cov(proj_one, rowvar=False)


        proj_X = np.dot(X, self.w)
        # Calculate the projected likelihood
        # likelihoods = np.array(list(
        #     [self.prior_zero*self.__post_probabilty(
        #         x, self.gaussian_mean_zero, self.gaussian_cov_zero, dim=self.num_dims),
        #      self.prior_one*self.__post_probabilty(
        #         x, self.gaussian_mean_one, self.gaussian_cov_one, dim=self.num_dims)
        #     ] for x in proj_X
        # ))
        likelihoods = np.array(list(
            [self.__post_probabilty(
                x, self.gaussian_mean_zero, self.gaussian_cov_zero, dim=self.num_dims),
             self.__post_probabilty(
                x, self.gaussian_mean_one, self.gaussian_cov_one, dim=self.num_dims)
            ] for x in proj_X
        ))

        y_pred = np.argmax(likelihoods, axis = 1)
        
        train_acc = np.mean(y_pred == y)
        print("Training accuracy: %f" %train_acc)

    '''
    Solve this problem
    '''
    def test_acc_gaussian(self, X, y):
        proj_X = np.dot(X, self.w)
        # Calculate the projected likelihood
        # likelihoods = np.array(list(
        #     [self.prior_zero*self.__post_probabilty(
        #         x, self.gaussian_mean_zero, self.gaussian_cov_zero, dim=self.num_dims),
        #      self.prior_one*self.__post_probabilty(
        #         x, self.gaussian_mean_one, self.gaussian_cov_one, dim=self.num_dims)
        #     ] for x in proj_X
        # ))
        likelihoods = np.array(list(
            [self.__post_probabilty(
                x, self.gaussian_mean_zero, self.gaussian_cov_zero, dim=self.num_dims),
             self.__post_probabilty(
                x, self.gaussian_mean_one, self.gaussian_cov_one, dim=self.num_dims)
            ] for x in proj_X
        ))

        y_pred = np.argmax(likelihoods, axis = 1)
        test_acc = np.mean(y_pred == y)
        print("Testing accuracy: %f" %test_acc)

    # calculate the probability density for gaussian distribution
    def __post_probabilty(self, data, mean, cov, dim=1):
        if dim == 1:
            cons = 1./(np.math.sqrt(2*np.math.pi*cov))
            return cons*np.exp(-(data - mean)**2/2*cov)
        else:
            cons = 1./((2*np.pi)**(len(data)/2.)*np.linalg.det(cov)**(-0.5))
            return cons*np.exp(-np.dot(np.dot((data-mean),np.linalg.inv(cov)),(data-mean).T)/2.)