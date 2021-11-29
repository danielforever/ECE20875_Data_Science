import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import preprocessing
import statistics
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import r2_score
clf = linear_model.LinearRegression()
from sklearn import metrics
import warnings
from sklearn.linear_model import LogisticRegression



def main():
    #Importing dataset
    diamonds = pd.read_csv('diamonds.csv')
    
    #Feature and target matrices
    X = diamonds[['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity', 'cut', 'color']]
    y = diamonds[['price']]
    x = ['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity', 'cut', 'color']
    X = normalize(X)
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)
    
    
    #####################################################
    #linear_model_from_sklearn
    clf.fit(np.array(X),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    print("coef:",coef)
    print(np.dot(coef[0], [0.25, 4, 3, 2, 60, 55,  5, 3, 3]))
    ###############################################
    
    ########################################
    #logistic_regression_from_sklearn
    #logreg = LogisticRegression()

    #logreg.fit(np.array(X).transpose(),y)
    #print(logreg.coef_)
    #print(np.dot(logreg.coef_, [0.25, 60, 55, 4, 3, 2, 5, 3, 3]))
    ########################################################
    
    MODEL = []
    MSE = []
    lmbda = list(range(1,100))
    for l in lmbda:
##        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)
            
##        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)
##
##        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)
        #print("MSE",MSE)
##    #Plot the MSE as a function of lmbda
    plt.title("MSE as lmbda")
    plt.xlabel("MSE")
    plt.ylabel("lmbda")
    plt.plot(lmbda,MSE,'-o')
    plt.show()
##
##    #Find best value of lmbda in terms of MSE
    mimmse = min(MSE)
    print("min",mimmse)
    
    ind = lmbda[MSE.index(mimmse)]-1#fill in
    print("index",ind)
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))
    print("[0.25, 60, 55, 4, 3, 2, 5, 3, 3] ceof",model_best.coef_)
    print(np.dot(model_best.coef_, [0.25, 4, 3, 2, 60, 55, 5, 3, 3]))
    return model_best


#Function that normalizes features to zero mean and unit variance.
#Input: Feature matrix X.
#Output: X, the normalized version of the feature matrix.

def normalize(X):
    for i in X:
##        print(i)
        Std =  np.std(X[i])
##        print(X[i][0])
        mean = statistics.mean(X[i])
        X[i]=[(j-mean)/Std for j in X[i]]
    return X


#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):
    model = linear_model.Ridge(alpha = l, fit_intercept = True)
    model.fit(X, y)

    return model
##
##
###Function that calculates the mean squared error of the model on the input dataset.
###Input: Feature matrix X, target variable vector y, numpy model object
###Output: mse, the mean squared error
def error(X,y,model):
    mse = mean_squared_error(y,model.predict(X))
    return mse
if __name__ == '__main__':
    main()
