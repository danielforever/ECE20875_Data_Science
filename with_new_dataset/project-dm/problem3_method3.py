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
from sklearn import linear_model
clf = linear_model.LinearRegression()
from sklearn.linear_model import LogisticRegression
#******************************************************************************
#This function produces dataset you want form the whole data.
#INPUT1: wholedata
#INPUT2: a list of data types        ex) ['Precipitation','Brooklyn Bridge'], ['High Temp (°F)','Brooklyn Bridge'] and etc.
#OUTPUT: list of data set            ex) [Precipitation, NumberOfBikes] = [[p1,p2,p3,p4...], [n1,n2,n3,n4...]]
def makeDataset(wholedata, bridge,result):
    df  = pd.read_csv(wholedata,header=None,names=['Date',' Day','High Temp (°F)','Low Temp (°F)','Precipitation','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'])
    df.replace(to_replace=['\s\(S\)'],value = '',inplace=True,regex=True)
    df.replace(to_replace=['T'],value = '0',inplace=True)
    df.replace(to_replace=[','],value = '',inplace=True,regex=True)

    
    df[result[0]] = df[result[0]][1:].astype(float)
##    print(df[result[0]])
##    df[result] = pd.to_numeric(df[result][1:])
##    print("df",df)
##    print(df[result[0]]>int(545))
    indexname = df[df[result[0]]>int(8000)].index
##    df.drop(indexname,inplace=True)
    indexname = df[df[result[0]]<int(1000)].index
##    df.drop(indexname,inplace=True)
##    print("df: ",df)
    dataset = []
    data=[]
    datasety=[]
    for i in bridge:
        for j in df[i][1:]:
            a=float(j)
            data.append(a)
            
        dataset.append(data)
        data=[]
    for j in result:
        for i in df[j][1:]:
            a = i
            if a >0:
                datasety.append(float(1))
            else:
                datasety.append(float(0))
##    print(dataset)
##    print(data)
    return dataset,datasety

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
##    print(x,y)
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
def bestModel(dataset,y):
    
    N = len(dataset[0])
    d = 0
    best_degree = 0
    flag = True
    tempMSE = math.inf
    
    while d <= 20:#flag:
        totalSE = 0
##        print("degree",d)
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
##            a.reverse()
##            print(a)
            result = np.dot(coefficients,a)
            # print(element1, result)
            totalSE = totalSE + sqError(element1, result)
        MSE = round(totalSE / N,0)
        
##        print(MSE)
##        print(tempMSE)
##        print(MSE <= tempMSE)
        if MSE <= tempMSE:
            best_degree = d 
            best_coefficient = coefficients
            tempMSE = MSE
        d += 1
##    print(tempMSE)
    return best_degree, best_coefficient

def accuraucyforlinear(coef,dataset,y,title):
    pre=[]
    ans=[]
    total=0
    dataset.append([float(1)]*len(dataset[0]))
    result = np.dot(np.array(coef),np.array(dataset))
    for i in range(len(dataset[0])):
        if result[i] <=0.5:
            if result[i] <0:
                result[i]=0
            ansx=0
        else:
            ansx=1
        
        if y[i]>0:
            ansy=1

        else:
            ansy=0
        if ansx is ansy:
            total=total+1
    plt.title(title)
    x1,y1 = zip(*sorted(zip( range(len(dataset[0])),y)))
    plt.plot(x1,y1)
    x2,y2 = zip(*sorted(zip( range(len(dataset[0])), result)))
    plt.plot(x2,y2)
    plt.legend(['dataset','prediction'])
    plt.suptitle('',fontsize=10)
    plt.xlabel('day')
    plt.ylabel('Precipitation')
    print(round(total/len(result),4))
    print("r:",r2_score(y,result))
    plt.show()
    plt.clf()
    return total/len(result)
    
##def predict(dataset,predict):
    
        
#******************************************************************************  
    
#******************************************************************************
#From this line, we can exam our program whether it works correctly.
if __name__ == '__main__':
##    print([2**j for j in range(5)].reverse())
    x,y	 = makeDataset('bike-data.csv', ['Total'],['Precipitation'])
##    print(x)
##    print(y)
##    d,coef = bestModel(x,y)
##    print(x)
##    print(y)
##    logreg = LogisticRegression()
##    logreg.fit(np.array(x).transpose(),y)
    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    
    print(coef)
##    coef = [-1257.2214768,71.3495114,-23.01125034,-752.5955258294061]
    accuraucyforlinear(coef,x,y,'Precipitation')
            #print(BrooklynBridgePrecipitation[1])
##    
##    a = [[100,200,300,400,500],[134.5,248.3,263.4,2783,736]]
####    pltScatter(BrooklynBridgePrecipitation)
##    
##    best_degree, best_coefficient = bestModel(BrooklynBridgePrecipitation)
####    print(best_degree, best_coefficient,best_degree)
##    print(accuraucy(best_coefficient,BrooklynBridgePrecipitation,best_degree))
##    #print(len(BrooklynBridgePrecipitation[0]))

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
