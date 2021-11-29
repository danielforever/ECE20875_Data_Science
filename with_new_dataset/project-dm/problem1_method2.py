import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import math
from sklearn.metrics import r2_score
from sklearn import linear_model
clf = linear_model.LinearRegression()

def makeDataset(wholedata, bridge,result):
    df  = pd.read_csv(wholedata,header=None,names=['Date',' Day','High Temp (°F)','Low Temp (°F)','Precipitation','Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'])
    df.replace(to_replace=['\s\(S\)'],value = '',inplace=True,regex=True)
    df.replace(to_replace=['T'],value = '0',inplace=True)
    df.replace(to_replace=[','],value = '',inplace=True,regex=True)
    
    df[result[0]] = df[result[0]][1:].astype(float)
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
            datasety.append(a)
    return dataset,datasety


def accuraucy(coef,dataset,y,title):
    pre=[]
    ans=[]
    total=0
    dataset.append([float(1)]*len(dataset[0]))
    result = np.dot(np.array(coef),np.array(dataset))
    for i in range(len(dataset[0])):
        total+=((abs(result[i]-y[i]))/y[i])*100

    plt.title(title)
    x1,y1 = zip(*sorted(zip( range(len(dataset[0])),y)))
    plt.plot(x1,y1)
    x2,y2 = zip(*sorted(zip( range(len(dataset[0])), result)))
    plt.plot(x2,y2)
    plt.legend(['dataset','prediction'])
    plt.suptitle('',fontsize=10)
    plt.xlabel('day')
    plt.ylabel('number of riders')
    print("accuracy:",round(100-total/len(result),4))
    plt.savefig(title)
    print("r:",round(r2_score(result,y),4))
    plt.clf()
    return total/len(result)
    
    

if __name__ == '__main__':

    x,y	 = makeDataset('bike-data.csv', ['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge'],['Total'])


    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    
    print("coef:",coef)
    accuraucy(coef,x,y,"without Queensboro Bridge")
    
    x,y	 = makeDataset('bike-data.csv', ['Brooklyn Bridge','Williamsburg Bridge','Queensboro Bridge'],['Total'])


    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    
    print("coef:",coef)
    accuraucy(coef,x,y,"without Manhattan Bridge")
    x,y	 = makeDataset('bike-data.csv', ['Brooklyn Bridge','Manhattan Bridge','Queensboro Bridge'],['Total'])


    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    
    print("coef:",coef)

    accuraucy(coef,x,y,"without Williamsburg Bridge")
    x,y	 = makeDataset('bike-data.csv', ['Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge'],['Total'])


    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    
    print("coef:",coef)
    accuraucy(coef,x,y,"without Brooklyn Bridge")
    
    x,y	 = makeDataset('bike-data.csv', ['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge'],['Total'])
    clf.fit(np.array(x).transpose(),y)
    coef = clf.coef_.tolist()
    coef.append(float(clf.intercept_))
    print("coef:",coef)
    accuraucy(coef,x,y,"Total")


