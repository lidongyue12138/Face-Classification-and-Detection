'''
This is the code for Model Construction: Logistic Model
TODO:
    1. build model
    2. train model
    3. test accuracy

    debug langevin dynamics
'''
# Necessary Imports
import numpy as np
import math


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def fit_gradient_based(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def fit_RGD(self, X, y, mini_batch_size = 500):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            '''
            Implement a RGD: mini batch gradient descent
            '''
            select_index = np.random.randint(y.size, size=mini_batch_size)

            X_mini_batch = X.T[:,select_index]
            gradient = np.dot(X_mini_batch, (h - y)[select_index]) / mini_batch_size
            self.theta -= self.lr * gradient
            ''' end '''
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

    def fit_langevin_dynamics(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            '''
            Implement a langevin dynamics: Stochastic Gradient Langevin dynamics
            '''
            gradient = np.dot(X.T, (h - y)) / y.size
            update = self.lr/2 * gradient + np.random.normal(loc=0, scale=self.lr)
            self.theta -= update
            ''' end '''
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \n')
                print("Training acc: %f" %(self.predict_accuracy(X, y, isTraining = True)))

    '''
    Used for prediction
    '''
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold = 0.5):
        return self.predict_prob(X) >= threshold

    def predict_accuracy(self, X, y, isTraining = False):
        if isTraining:
            y_pred = self.__sigmoid(np.dot(X, self.theta)) >= 0.5
        else:
            y_pred = self.predict(X)
        return np.mean(y == y_pred)

    '''
    Helper functions
    '''
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()