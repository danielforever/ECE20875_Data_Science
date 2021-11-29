#This is the program file that includes all program that we use for this project.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from itertools import islice
import warnings
#******************************************************************************
#This function produces dataset you want form the whole data.
#INPUT1: wholedata
#INPUT2: a list of data types        ex) ['Precipitation','Brooklyn Bridge'], ['High Temp (°F)','Brooklyn Bridge'] and etc.
#OUTPUT: list of data set            ex) [Precipitation, NumberOfBikes] = [[p1,p2,p3,p4...], [n1,n2,n3,n4...]]
def makeDataset(wholedata, bridge):
    df  = pd.read_csv(wholedata,header=None,names=['number','Date',' Day','High Temp (Â°F)','Low Temp (Â°F)','Precipitation','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'])
    df.replace(to_replace=['\s\(S\)'],value = '',inplace=True,regex=True)
    df.replace(to_replace=['T'],value = '0',inplace=True)
    dataset = []
    data=[]
    for i in bridge:
        for j in df[i][1:]:
            data.append(float(j))
        dataset.append(data)
        data=[]
    #重要事項！！　'Precipitation' has (S) and T. (S) means snow so, just make it only value in other words reduce (S) from the data
    #重要事項！！　T means Trace. Therefore, make T as 0. #check
    
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

    ax = fig.add_subplot(1,1,1)

    ax.scatter(x,y)

    ax.set_title('scatter plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.show()
    
#******************************************************************************
    
#******************************************************************************
def bestModel(dataset):
    
    N = len(dataset[0])
    d = 0
    while d <= 10:
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
            result = np.dot(coefficients,[element0**j for j in range(d+1)])
            totalSE = totalSE + sqError(element1, result)
        MSE = totalSE / N
        print(MSE, ' + @ degree = +' , d)
        d += 1
        
#******************************************************************************
    
#******************************************************************************
#From this line, we can exam our program whether it works correctly.
if __name__ == '__main__':
    
    BrooklynBridgePrecipitation	 = makeDataset('nyc-east-river-bicycle-counts.csv', ['Precipitation','Brooklyn Bridge'])
    #print(BrooklynBridgePrecipitation[1])
    
    a = [[100,200,300,400,500],[134.5,248.3,263.4,2783,736]]
    pltScatter(BrooklynBridgePrecipitation)
    
    bestModel(BrooklynBridgePrecipitation)
    
    #print(len(BrooklynBridgePrecipitation[0]))

    '''
    for i in range(len(BrooklynBridgePrecipitation[0])):
        temp = BrooklynBridgePrecipitation
        element0 = temp[0][i]
        element1 = temp[1][i]
        del temp[0][i]
        del temp[1][i]
        coefficients = makeReg(temp, 1)
        result = np.dot(coefficients,[temp[0][i]**j for j in range(1+1)])
        temp[0].insert(i, element0)
        temp[1].insert(i, element1)
        print(i)
     
    temp = BrooklynBridgePrecipitation
    element0 = temp[0][1]
    element1 = temp[1][1]
    del temp[0][1]
    del temp[1][1]
    #coefficients = makeReg(temp, 1)
    #result = np.dot(coefficients,[temp[0][209]**j for j in range(1+1)])
    temp[0].insert(1, element0)
    temp[1].insert(1, element1)
    print(temp)
   '''
