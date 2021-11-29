#This is the program file that includes all program that we use for this project.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from itertools import islice
import warnings
import math
from sklearn import linear_model
from sklearn.metrics import r2_score
#******************************************************************************
#This function produces dataset you want form the whole data.
#INPUT1: wholedata
#INPUT2: a list of data types        ex) ['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'].
#OUTPUT: list of data set            ex) [Precipitation, NumberOfBikes] = [[p1,p2,p3,p4...], [n1,n2,n3,n4...]]
def makeDataset(wholedata, bridge):
    df  = pd.read_csv(wholedata,header=None,names=['Date',' Day','High Temp (°F)','Low Temp (°F)','Precipitation','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'])
    flag=0;
    dataset = []
    data=[]
    for i in bridge:
        for j in df[i][1:]:
            a=j.replace(',', '')
            if flag==1:
                data.append(float(a)/4)          
            else:
                data.append(float(a))
            
        dataset.append(data)
        flag=1
        data=[]

    return dataset

#******************************************************************************
    
#******************************************************************************
#This function returns regression results.
#INPUT1: data set                                    ex) [Precipitation, NumberOfBikes] = [[p1,p2,p3,p4...], [n1,n2,n3,n4...]]
#INPUT2: the dgree of linear regression equation     ex) if d = 3, y = a0 + a1*x + a2*x^2 +a3*x^3
#OUTPUT: list of coefficients                        ex) [ad,a(d-1),a(d-2),...,a0]
def makeReg(data, degree):
    coefficients = []
    x = data[0]
    y = data[1]
    coefficients = list(np.polyfit(x, y, degree))
    warnings.filterwarnings("ignore")
    
    coefficients.reverse()
    
    return coefficients

#******************************************************************************
    
#******************************************************************************
#This function calculate square error.
#INPUT1: value1
#INPUT2: value2
#OUTPUT: MSE
def sqError(value1, value2):
    return (value1 - value2)**2
    
#******************************************************************************
    
#******************************************************************************
#This function plots scatter shart.
#INPUT: dataset
def pltScatter(dataset):
    
    # generate data
    x = np.array(dataset[0])
    y = np.array(dataset[1])

    fig = plt.figure()
    warnings.filterwarnings("ignore")
    
    ax = fig.add_subplot(1,1,1)
    warnings.filterwarnings("ignore")
    
    ax.scatter(x,y)

    ax.set_title('scatter plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    warnings.filterwarnings("ignore")
    fig.show()
    
#******************************************************************************
    
#******************************************************************************
def bestModel(dataset):
    
    N = len(dataset[0])
    d = 0
    best_degree = 0
    flag = True
    tempMSE = math.inf
    
    
    while d <= 20:
        totalSE = 0
        for i in range(N):
            temp = dataset
            element0 = temp[0][i]
            element1 = temp[1][i]
            del temp[0][i]
            del temp[1][i]
            coefficients = makeReg(temp, d)
            temp[0].insert(i, element0)
            temp[1].insert(i, element1)
            a = [element0**j for j in range(d+1)]
            result = np.dot(coefficients,a)
            totalSE = totalSE + sqError(element1, result)
        MSE = round(totalSE / N,0)
        if MSE <= tempMSE:
            best_degree = d 
            best_coefficient = coefficients
            tempMSE = MSE
            if MSE == 0:
                break
        d += 1
    return best_degree, best_coefficient

def accuraucy(coef,dataset,d):
    pre=[]
    ans=[]
    total=0
    print("best degree",d)
    for i in dataset[0]:
        result = np.dot(coef,[i**j for j in range(d+1)])
        
        pre.append(result)
    for i in range(len(pre)):
        total+=((abs(pre[i]-dataset[1][i]))/dataset[1][i])*100
    x1,y1 = zip(*sorted(zip( dataset[0],pre)))
    fig = plt.figure()
    

    plt.plot(x1,y1)
    x2,y2 = zip(*sorted(zip( dataset[0], dataset[1])))
    plt.plot(x2,y2)
    plt.suptitle('The model is trained only by Precipitation',fontsize=10)
    plt.xlabel('The amount of rain fall(mm)')
    plt.ylabel('Person')
    plt.ylim(0, 8000)
    plt.show()
    print("r:",r2_score(dataset[1],pre))
    print("coef:",coef)
    return total/len(pre)
    
    
        
#******************************************************************************
    
#******************************************************************************
#From this line, we can exam our program whether it works correctly.
if __name__ == '__main__':
    datasetB  = makeDataset('bike-data.csv', ['Brooklyn Bridge','Total'])
    print(bestModel(datasetB))
    d,c = bestModel(datasetB)
    accuraucy(c,datasetB,d)
    datasetB  = makeDataset('bike-data.csv', ['Manhattan Bridge','Total'])
    print(bestModel(datasetB))
    d,c = bestModel(datasetB)
    accuraucy(c,datasetB,d)
    datasetB  = makeDataset('bike-data.csv', ['Williamsburg Bridge','Total'])
    print(bestModel(datasetB))
    d,c = bestModel(datasetB)
    accuraucy(c,datasetB,d)
    datasetB  = makeDataset('bike-data.csv', ['Queensboro Bridge','Total'])
    print(bestModel(datasetB))
    d,c = bestModel(datasetB)
    accuraucy(c,datasetB,d)

