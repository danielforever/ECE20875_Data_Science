import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D
#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of n = degrees[i].
def main(datapath, degrees):
    paramFits = []
    org = []
    x=[]
    y=[]
    dic={}
    y_ans=0
    with open(datapath) as f:
        k=0
        for i in f.readlines():
            a = i.split("\t")
##            print(a[0],a[1])
            a[1] = a[1][0:-2]
            b = np.array([[a[0]],[a[1]]])
            org.append([])
            for j in range(0,2):
                org[k].append(a[j])
            k=k+1
    for i in org:
        x.append(float(i[0]))
        y.append(float(i[1]))
    colors = ["red", "green", "blue", "orange", "black"]
    groups = ["N1","N2","N3","N4","N5"]
    for i in degrees:
        print(len(degrees))
        X = feature_matrix(x,i)
##        print("origX",X)
        B = least_squares(X, y)
        paramFits.append(B)
        x1 = x
        y1 = []
##        print("x1",x1)
####        print("B",B)
        for k in range(len(x1)):
            for j in range (i+1):
                y_ans = y_ans + (x1[k]**j)*B[i-j]
            y1.append(y_ans)
##            print("y1",y1)
            y_ans=0
        plt.xlabel("X-axis")    
        plt.ylabel("Y-axis")
        plt.title("dataset")
##        plt.scatter(x,y,s=area, c=colors, alpha=0.5)
        x1,y1 = zip(*sorted(zip(x1, y1)))
        line1, =plt.plot(x1,y1,marker='o',label=groups[i-1])
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
##        plt.legend()
    plt.show()
        
    
    
    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    #in problem1.txt.
    #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    #for the model parameters in each case. Append the result to paramFits each time.

    return paramFits
###Return the feature matrix for fitting a polynomial of degree n based on the explanatory variable
###samples in x.
###Input: x as a list of the independent variable samples, and n as an integer.
###Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
###for the ith sample. Viewed as a matrix, X should have dimension #samples by n+1.
def feature_matrix(x, n):
    X=[]
    num=[]
    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^n, x^(n-1), ..., x^0.
##    regr = linear_model.LinearRegression(fit_intercept=True)
    for i in range(0,n+1):
        num.append(n-i)
##    print(num)
    for j in range (0,len(x)):
##        print(x[0][j])
        X.append([])
        for i in num:
            X[j].append(float(x[j])**i)
            
    return X
##
##
###Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
###Input: X as a list of features for each sample, and y as a list of target variable samples.
###Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)
    t1 = np.dot(np.transpose(X),X)
    t2 = np.dot(np.linalg.inv(t1),np.transpose(X))
    B = np.dot(t2,y)

    #fill in
    #Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.

    return B

if __name__ == '__main__':
    datapath = 'poly.txt'
    degrees = [1,2,3,4,5]
##    main(datapath, degrees)
    paramFits = main(datapath, degrees)
    print(paramFits)
